import sys, os
from pathlib import Path
import torch
import torchaudio

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vq_vae import SegmentVQVAE, chunk_encodec_tokens
from models.audio_encoder import AudioEncoder

# ─────────────────────────── config ───────────────────────────
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
AUDIO_IN      = "dataset/song_0001/song_001.mp3"
CHECKPOINT = "checkpoints/vqvae_dataset.pt"
SEG_LEN       = 256  # EnCodec frames per segment
OUTPUT_WAV    = "reconstructed_vqvae.wav"

# ─────────────────────────── load audio encoder ────────────────
audio_encoder = AudioEncoder(device=DEVICE, sample_rate=24000, bandwidth=6.0)
tokens = audio_encoder.encode(AUDIO_IN).to(DEVICE)  # (T_full, C)
T_full, C = tokens.shape

# ───────────────────────── chunk into segments ─────────────────
segments = chunk_encodec_tokens(tokens, seg_len=SEG_LEN)  # list of (SEG_LEN, C)
num_segs = len(segments)

# ─────────────────────────── load model ─────────────────────────
# instantiate VQ-VAE
V = audio_encoder.vocab_size
VOCAB_SIZE = audio_encoder.vocab_size
model = SegmentVQVAE(
    vocab_size=VOCAB_SIZE,
    n_codebooks=C,
    block_pairs=16,
    seg_len=SEG_LEN,
    latent_dim=256,
    emb_dim=256
).to(DEVICE)

# load checkpoint
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ───────────────────────── reconstruct tokens ───────────────────
recon_segments = []
with torch.no_grad():
    for i in range(num_segs):
        # get prev, curr, next with edge padding
        prev = segments[i - 1] if i > 0        else segments[0]
        curr = segments[i]
        nxt  = segments[i + 1] if i < num_segs - 1 else segments[-1]

        # to batch
        prev = prev.unsqueeze(0).to(DEVICE)  # [1, SEG_LEN, C]
        curr = curr.unsqueeze(0).to(DEVICE)
        nxt  = nxt.unsqueeze(0).to(DEVICE)

        # encode → quantize
        p_i, p_v = model.encoder(prev)
        c_i, c_v = model.encoder(curr)
        n_i, n_v = model.encoder(nxt)


        # decode logits
        logits = model.decoder(
            p_i, p_v,
            c_i, c_v,
            n_i, n_v
        )  # [1, SEG_LEN, C, V]

        # take argmax over vocab to get token IDs
        tokens_pred = logits.argmax(dim=-1).squeeze(0).cpu()  # (SEG_LEN, C)
        recon_segments.append(tokens_pred)

# flatten back to full sequence length (may pad at end)
recon_tokens = torch.cat(recon_segments, dim=0)  # (num_segs * SEG_LEN, C)
recon_tokens = recon_tokens[:T_full]             # trim to original length

# ─────────────────────────── decode audio ───────────────────────
wave = audio_encoder.decode(recon_tokens.to(DEVICE))  # (C_out, T_samples)
# AudioEncoder.decode returns (C_out, T_samples); adjust if shape differs
sample_rate = audio_encoder.sample_rate

# make sure output dir exists
os.makedirs(Path(OUTPUT_WAV).parent, exist_ok=True)
# save
torchaudio.save(OUTPUT_WAV, wave, sample_rate)
print(f"✅ Reconstructed audio saved to {OUTPUT_WAV}")

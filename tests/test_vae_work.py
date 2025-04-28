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
CHECKPOINT    = "checkpoints/vqvae_dataset.pt"
SEG_LEN       = 256  # frames per segment
OUTPUT_WAV    = "reconstructed_ae.wav"

# ─────────────────────────── load audio encoder ────────────────
audio_encoder = AudioEncoder(device=torch.device("cpu"), sample_rate=48000)  # ← match training config
tokens = audio_encoder.encode(AUDIO_IN).to(DEVICE)  # (T_full, C)
T_full, C = tokens.shape

# ───────────────────────── chunk into segments ─────────────────
segments = chunk_encodec_tokens(tokens, seg_len=SEG_LEN)  # list of (SEG_LEN, C)
num_segs = len(segments)

# ─────────────────────────── load model ─────────────────────────
model = SegmentVQVAE(
    input_dim=C,
    latent_dim=1024,  # match your training config
    seq_len=SEG_LEN
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
        prev = segments[i - 1] if i > 0 else segments[0]
        curr = segments[i]
        nxt  = segments[i + 1] if i < num_segs - 1 else segments[-1]

        # to batch
        prev = prev.unsqueeze(0).to(DEVICE)  # [1, SEG_LEN, C]
        curr = curr.unsqueeze(0).to(DEVICE)
        nxt  = nxt.unsqueeze(0).to(DEVICE)

        # encode
        z_prev = model.encoder(prev)
        z_curr = model.encoder(curr)
        z_next = model.encoder(nxt)

        # decode
        recon = model.decoder(z_prev, z_curr, z_next)  # (1, SEG_LEN, C)

        recon_segments.append(recon.squeeze(0).cpu())  # (SEG_LEN, C)

# flatten back to full sequence length
recon_tokens = torch.cat(recon_segments, dim=0)  # (num_segs * SEG_LEN, C)
recon_tokens = recon_tokens[:T_full]             # trim to original length

# ─────────────────────────── decode audio ───────────────────────
# Now, decode from reconstructed tokens
wave = audio_encoder.decode(recon_tokens.to(DEVICE))  # (C_out, T_samples)
print(wave)
sample_rate = audio_encoder.sample_rate

# make sure output dir exists
os.makedirs(Path(OUTPUT_WAV).parent, exist_ok=True)

# save
torchaudio.save(OUTPUT_WAV, wave, sample_rate)
print(f"✅ Reconstructed audio saved to {OUTPUT_WAV}")

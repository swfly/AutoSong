import sys, os
from pathlib import Path
import torch
import torchaudio
import matplotlib.pyplot as plt

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.autoencoder import SegmentAutoEncoder, SpectrogramDiscriminator
from utils.chunk_segments import chunk_segments
from models.audio_encoder import AudioEncoder

# ─────────────────────────── config ───────────────────────────
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
AUDIO_IN      = "dataset/song_0001/accompaniment.wav"
CHECKPOINT    = "checkpoints/vqvae_dataset.pt"
SEG_LEN       = 256  # frames per segment
OUTPUT_WAV    = "reconstructed_ae.wav"

# ─────────────────────────── load audio encoder ────────────────
encoder = AudioEncoder(device=torch.device("cpu")) 
tokens = encoder.encode(AUDIO_IN).to(device)  # (T_full, C)
T_full, C = tokens.shape

# ───────────────────────── chunk into segments ─────────────────
segments = chunk_segments(tokens, seg_len=SEG_LEN)  # list of (SEG_LEN, C)
num_segs = len(segments)

# ─────────────────────────── load model ─────────────────────────

model = SegmentAutoEncoder.create(device, SEG_LEN,encoder.dim)

# load checkpoint
ckpt = torch.load(CHECKPOINT, map_location=device)
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
        prev = prev.unsqueeze(0).to(device)  # [1, SEG_LEN, C]
        curr = curr.unsqueeze(0).to(device)
        nxt  = nxt.unsqueeze(0).to(device)

        # encode
        z_prev = model.encoder(prev)
        z_curr = model.encoder(curr)
        z_next = model.encoder(nxt)

        z = torch.concat([z_prev, z_curr, z_next], dim=1)  # (B, latent_dim * 3)
        
        # decode
        recon = model.decoder(z)  # (1, SEG_LEN, B)
        recon_segments.append(recon.squeeze(0).cpu())  # (SEG_LEN, C)

# flatten back to full sequence length
recon_tokens = torch.cat(recon_segments, dim=0)  # (num_segs * SEG_LEN, C)
recon_tokens = recon_tokens[:T_full]             # trim to original length
print(f"✅ Reconstructed")
# ─────────────────────────── decode audio ───────────────────────
# Now, decode from reconstructed tokens
wave = encoder.decode(recon_tokens.to(device)[:2048])
# wave = wave.squeeze(0).detach().cpu().numpy()
# wave2 = encoder.decode(tokens)
# wave2 = wave2.squeeze(0).detach().cpu().numpy()
plt.plot(wave.squeeze(0).detach().cpu().numpy(),label="recons")
# plt.plot(wave2, label="original")
plt.legend()
plt.show()
# make sure output dir exists
os.makedirs(Path(OUTPUT_WAV).parent, exist_ok=True)

# save
torchaudio.save(OUTPUT_WAV, wave, encoder.sample_rate)
print(f"✅ Reconstructed audio saved to {OUTPUT_WAV}")

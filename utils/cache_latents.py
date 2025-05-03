import os
import sys
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.audio_encoder import AudioEncoder
from models.autoencoder import SegmentAutoEncoder
from utils.chunk_segments import chunk_segments

# ───────────── Config ─────────────
DATASET_DIR  = "dataset"
CHECKPOINT   = "checkpoints/vqvae_dataset.pt"
SEG_LEN      = 256
LATENT_PATH  = "cached_latents.pt"

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ───────────── Load Models ─────────────
print("🔧 Loading models...")
encoder = AudioEncoder(device=torch.device("cpu"))
model = SegmentAutoEncoder(
    input_dim=encoder.dim, latent_size=(32,32), latent_channels=4,
    network_channel_base=32, seq_len=SEG_LEN
).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ───────────── Process All Songs ─────────────
song_dirs = [
    os.path.join(DATASET_DIR, d)
    for d in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("song_")
]

print(f"🎼 Found {len(song_dirs)} songs in {DATASET_DIR}")
for song_path in tqdm(song_dirs, desc="Caching latents"):
    out_path = os.path.join(song_path, LATENT_PATH)
    # if os.path.exists(out_path):
    #     continue  # already cached

    try:
        audio_file = next(f for f in os.listdir(song_path) if f.endswith((".mp3", ".wav", ".flac")))
        audio_path = os.path.join(song_path, audio_file)
    except StopIteration:
        print(f"⚠️  No audio file in {song_path}")
        continue

    tokens = encoder.encode(audio_path).to(DEVICE)
    segments = chunk_segments(tokens, SEG_LEN)
    if len(segments) < 1:
        continue

    latents = []
    with torch.no_grad():
        for i in range(len(segments)):
            curr = segments[i]

            z = model.encoder(curr.unsqueeze(0).to(DEVICE)).squeeze(0)
            latents.append(z.cpu())

    # save latent list
    torch.save(latents, out_path)

print("✅ Done caching all latents.")

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
LATENT_EXT   = ".pt"

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ───────────── Load Models ─────────────
print("🔧 Loading models...")
encoder = AudioEncoder(device=torch.device("cpu"))

model = SegmentAutoEncoder(
    input_dim=encoder.dim, latent_size=(32,32), latent_channels=2,
    network_channel_base=32, seq_len=SEG_LEN
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ───────────── Process All Songs ─────────────
song_dirs = [
    os.path.join(DATASET_DIR, d)
    for d in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("song_")
]

print(f"🎼 Found {len(song_dirs)} song directories in {DATASET_DIR}")
for song_path in tqdm(song_dirs, desc="Caching latents"):
    audio_files = [f for f in os.listdir(song_path) if f.endswith((".mp3", ".wav", ".flac"))]
    
    if not audio_files:
        print(f"⚠️  No audio files in {song_path}")
        continue

    for audio_file in audio_files:
        audio_path = os.path.join(song_path, audio_file)
        out_path = os.path.splitext(audio_path)[0] + LATENT_EXT

        tokens = encoder.encode(audio_path).to(device)
        segments = chunk_segments(tokens, SEG_LEN)
        if len(segments) < 1:
            continue

        latents = []
        with torch.no_grad():
            for i in range(len(segments)):
                curr = segments[i]
                z = model.encoder(curr.unsqueeze(0).to(device)).squeeze(0)
                latents.append(z.cpu())

        torch.save(latents, out_path)

print("✅ Done caching all latents.")

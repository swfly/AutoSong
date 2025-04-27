import os
import sys
import random
import gc
import torch
import torch.nn.functional as F
from torch import nn, optim
import re

import math
from typing import List, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.audio_encoder import AudioEncoder
from models.vq_vae import TransformerAutoencoder  # <-- Your new AE


# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------

def chunk_encodec_tokens(tokens: torch.Tensor, seg_len: int) -> List[torch.Tensor]:
    """Split an EnCodec token grid (T, C) into equal-length segments (zero-padded)."""
    T, C = tokens.shape
    num_seg = math.ceil(T / seg_len)
    pad = seg_len * num_seg - T
    if pad:
        pad_tensor = torch.zeros(pad, C, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, pad_tensor], 0)
    return list(tokens.view(num_seg, seg_len, C))

# ─────────────────────── config ─────────────────────── #
SEG_LEN          = 256
BATCH_SIZE       = 16
EPOCHS           = 100000
CHECKPOINT_PATH  = "checkpoints/discrete_ae_dataset.pt"
DATASET_DIR      = "dataset"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

# ─────────────────── helper: dataset listing ─────────────────── #
def get_song_list(dataset_dir, max_songs=None):
    max_songs = 1
    def song_number(s):
        m = re.search(r"song_(\d+)", s)
        return int(m.group(1)) if m else float("inf")

    all_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("song_")
    ]
    sorted_dirs = sorted(all_dirs, key=song_number)
    return sorted_dirs if max_songs is None else sorted_dirs[:max_songs]

# ─────────────────── helper: load triplets ─────────────────── #
def get_triplets_from_song(path: str):
    cache_path = os.path.join(path, "cached_encoding.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        tokens = data["tokens"]
    else:
        audio_file = next(f for f in os.listdir(path) if f.endswith((".mp3", ".flac", ".wav")))
        audio_path = os.path.join(path, audio_file)
        tokens = encoder.encode(audio_path).cpu()
        torch.save({"tokens": tokens}, cache_path)
        print(f"💾 Saved cache to {cache_path}")
    segments = chunk_encodec_tokens(tokens, SEG_LEN)
    return [(segments[i], segments[i+1], segments[i+2]) for i in range(len(segments) - 2)] if len(segments) >= 3 else []

# ──────────────────── initialize model & encoder ──────────────────── #
encoder = AudioEncoder(device=device, sample_rate=24000, bandwidth=6.0)
VOCAB_SIZE = encoder.vocab_size

# Load a sample to determine codebook count
tmp_tokens = encoder.encode(os.path.join(DATASET_DIR, "song_001", "song_001.mp3"))
N_CODEBOOKS = tmp_tokens.shape[1]

model = TransformerAutoencoder(
    input_token_vocab_size=VOCAB_SIZE,
    n_codebooks=N_CODEBOOKS,
    segment_length=SEG_LEN,
    latent_seq_len=256,
    latent_vocab_size=2048,
    embed_dim=256,
    latent_dim=256,
    num_layers=2,
    num_heads=4,
    dropout=0.0
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
start_epoch = 1

if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    #optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"⏩ Resuming from epoch {start_epoch}")

song_list = get_song_list(DATASET_DIR, max_songs=1)

# ─────────────────────── training loop ─────────────────────── #
print("🚀 Starting Transformer AE training …")
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()

    # ───── Sample triplets
    triplets = []
    while len(triplets) < BATCH_SIZE:
        song_dir = random.choice(song_list)
        parts = get_triplets_from_song(os.path.join(DATASET_DIR, song_dir))
        if parts:
            take = min(len(parts), BATCH_SIZE - len(triplets), BATCH_SIZE // 4)
            triplets.extend(random.sample(parts, take))

    # ───── Stack batches
    batch_curr = torch.stack([t[1] for t in triplets]).to(device)

    # ───── Forward
    optimizer.zero_grad()
    out = model(batch_curr)  # tokens: [B, T, C]
    loss = out["loss"]
    loss.backward()
    optimizer.step()

    # ───── Logging
    print(f"[Epoch {epoch:04d}] Loss: {loss.item():.4f}")

    # ───── Save checkpoint
    if epoch % 50 == 0 or epoch == EPOCHS:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"    💾 Saved checkpoint → {CHECKPOINT_PATH}")

    del batch_curr
    torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache()
    gc.collect()

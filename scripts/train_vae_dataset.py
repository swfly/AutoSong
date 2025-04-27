# 📄 scripts/train_vae_dataset.py

import os, sys, random, gc
import torch
import torch.nn.functional as F
from torch import nn, optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.audio_encoder import AudioEncoder
from models.vq_vae import SegmentVQVAE, chunk_encodec_tokens
import os
import re

def get_song_list(dataset_dir="dataset", max_songs=None):
    def song_number(s):
        match = re.search(r"song_(\d+)", s)
        return int(match.group(1)) if match else float("inf")

    all_dirs = [d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("song_")]

    sorted_dirs = sorted(all_dirs, key=song_number)
    if max_songs is not None:
        sorted_dirs = sorted_dirs[:max_songs]

    return sorted_dirs

def get_triplets_from_song(path: str) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load one song and return a list of (prev, curr, next) triplets.
    """
    cache_path = os.path.join(path, "cached_encoding.pt")
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        tokens = data["tokens"]
    else:
        audio_file = next(f for f in os.listdir(path) if f.endswith((".mp3", ".flac")))
        audio_path = os.path.join(path, audio_file)
        tokens = encoder.encode(audio_path).cpu()
        torch.save({"tokens": tokens}, cache_path)
        print(f"💾 Saved cache to {cache_path}")

    segments = chunk_encodec_tokens(tokens, SEG_LEN)
    if len(segments) < 3:
        return []
    return [(segments[i], segments[i+1], segments[i+2]) for i in range(len(segments) - 2)]

# ─────────────────────── config ─────────────────────── #
SEG_LEN = 256
BATCH_SIZE = 64
EPOCHS = 100000
CHECKPOINT_PATH = "checkpoints/vqvae_dataset.pt"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

DATASET_DIR = "dataset"
encoder = AudioEncoder(device=device, sample_rate=24000, bandwidth=6.0)
VOCAB_SIZE = encoder.vocab_size

# Load a sample just to get codebook count
tmp_tokens = encoder.encode("dataset/song_001/song_001.mp3")
N_CODEBOOKS = tmp_tokens.shape[1]

model = SegmentVQVAE(
    vocab_size=VOCAB_SIZE,
    n_codebooks=N_CODEBOOKS,
    block_pairs=16,
    seg_len=SEG_LEN,
    latent_dim=256,
    emb_dim=256,
    num_codes=512,
    beta=0.2
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
start_epoch = 1

if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"⏩ Resuming from epoch {start_epoch}")

# ─────────────────────── training loop ─────────────────────── #
print("🚀 Starting full dataset VAE training …")
n_songs = 1
song_list = get_song_list(max_songs=n_songs)
train_vq = False
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    triplets = []
    print("learning with", n_songs,"songs")
    while len(triplets) < BATCH_SIZE:
        song_dir = random.choice([
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])
        song_dir=random.sample(song_list, 1)[0]
        song_path = os.path.join(DATASET_DIR, song_dir)
        song_triplets = get_triplets_from_song(song_path)
        if song_triplets:
            triplets.extend(random.sample(song_triplets, min(int(BATCH_SIZE / 4), min(len(song_triplets), BATCH_SIZE - len(triplets)))))

    batch_prev, batch_curr, batch_next = zip(*triplets)
    batch_prev = torch.stack(batch_prev).to(device)
    batch_curr = torch.stack(batch_curr).to(device)
    batch_next = torch.stack(batch_next).to(device)

    optimizer.zero_grad()
    if train_vq or True:
        loss, metrics = model(
            batch_prev, batch_curr, batch_next,
            zero_second_prob=0.2,
        )
    else:
        loss, metrics = model.train_ae(
            batch_prev, batch_curr, batch_next,
            zero_second_prob=0.2,
        )
    loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch:04d}] Loss: {metrics['total']:.4f} | Recon: {metrics['recon_loss']:.4f} | VQ: {metrics['vq_loss']:.4f}")
    if metrics['recon_loss'] < 5e-1:
        if not train_vq:
            train_vq = True
        else:
            n_songs += 2
            song_list = get_song_list(max_songs=n_songs)
            n_songs = min(len(song_list), n_songs)
    if epoch % 200 == 0 or epoch == EPOCHS:
        with torch.no_grad():
            z_i, z_v = model.encoder(batch_curr)
            _, code_ids_i, _ = model.vq_instru(z_i)
            _, code_ids_v, _ = model.vq_vocal(z_v)
            print(f"🧱 Instr codes used: {code_ids_i.unique().numel()}, Vocal codes used: {code_ids_v.unique().numel()}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

    del batch_prev, batch_curr, batch_next
    torch.mps.empty_cache()
    gc.collect()
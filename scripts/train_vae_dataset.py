# 📄 scripts/train_vae_dataset.py

import os, sys, random, gc
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.audio_encoder import AudioEncoder
# from models.vq_vae import SegmentVAE, SpectrogramDiscriminator
from models.autoencoder import SegmentAutoEncoder, SpectrogramDiscriminator
from utils.chunk_segments import chunk_segments
import os
import re
from typing import List, Tuple

# ───────────────────────── utility ────────────────────────────
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
        audio_file = next(f for f in os.listdir(path) if f.endswith(("wav", ".mp3", ".flac")))
        audio_path = os.path.join(path, audio_file)
        tokens = encoder.encode(audio_path).cpu()
        torch.save({"tokens": tokens}, cache_path)
        print(f"💾 Saved cache to {cache_path}")
    
    segments = chunk_segments(tokens, SEG_LEN)
    if len(segments) < 3:
        return []
    return [(segments[i], segments[i+1], segments[i+2]) for i in range(3, len(segments) - 3)]


# ─────────────────────── setups ─────────────────────── #
DATASET_DIR        = "dataset"
DATASET_INST_DIR   = "dataset_inst"
DATASET_VOCAL_DIR  = "dataset_vocal"
INST_RATE          = 0.2  # 30% epochs use instrumental-only data
VOCAL_RATE         = 0.2  # 20% epochs use vocal-only data

normal_song_list = get_song_list(DATASET_DIR, max_songs=10000)
inst_song_list   = get_song_list(DATASET_INST_DIR)
vocal_song_list  = get_song_list(DATASET_VOCAL_DIR)
n_songs = len(normal_song_list)
# n_songs = 1
# ─────────────────────── config ─────────────────────── #
SEG_LEN = 256
BATCH_SIZE = 16
EPOCHS = 100000
CHECKPOINT_PATH = "checkpoints/vqvae_dataset.pt"

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

encoder = AudioEncoder(device=torch.device("cpu"), sample_rate=48000)

# Load a sample just to get codebook count
tmp_tokens = get_triplets_from_song("dataset/song_001")[0][0]

model = SegmentAutoEncoder(
    input_dim=encoder.dim, latent_size=(32,32), latent_channels=4,
    network_channel_base=32, seq_len= SEG_LEN
).to(device)
discriminator = SpectrogramDiscriminator(1, base_dim = 12).to(device)


# model = SimpleLinearAE(input_dim=encoder.dim,latent_dim=1024,seq_len=SEG_LEN).to(device)
def build_optimizer_and_scheduler(model, base_lr=1e-4, warmup_steps=1000, total_steps=500_000):
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=base_lr, weight_decay=1e-5)
    # Warmup scheduler
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda)
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    return optimizer, optimizer_D, scheduler, scheduler_D


optimizer, optimizer_D, scheduler, scheduler_D = build_optimizer_and_scheduler(
    model, base_lr = 1e-4, warmup_steps=32, total_steps=EPOCHS)
start_epoch = 1

if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"⏩ Resuming from epoch {start_epoch}")

# ─────────────────────── training loop ─────────────────────── #
print("🚀 Starting full dataset VAE training …")
criterion = nn.BCELoss()
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    triplets = []
    r = random.random()
    mask_vocal = False
    mask_inst = False
    if r < VOCAL_RATE and vocal_song_list:
        mode = "vocal"
        mask_inst = True
        song_list = vocal_song_list
        base_dir = DATASET_VOCAL_DIR
    elif r < VOCAL_RATE + INST_RATE and inst_song_list:
        mode = "instr"
        mask_vocal = True
        song_list = inst_song_list
        base_dir = DATASET_INST_DIR
    else:
        mode = "mix"
        song_list = normal_song_list[:n_songs]
        base_dir = DATASET_DIR
    print(f"[Epoch {epoch:04d}] Mode: {mode.upper():>6} | songs: {len(song_list)}")
    while len(triplets) < BATCH_SIZE:
        song_dir = random.choice([
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])
        song_dir=random.sample(song_list, 1)[0]
        song_path = os.path.join(base_dir, song_dir)
        song_triplets = get_triplets_from_song(song_path)
        if song_triplets:
            triplets.extend(random.sample(song_triplets, min(int(BATCH_SIZE / 4), min(len(song_triplets), BATCH_SIZE - len(triplets)))))

    batch_prev, batch_curr, batch_next = zip(*triplets)
    batch_prev = torch.stack(batch_prev).to(device)
    batch_curr = torch.stack(batch_curr).to(device)
    batch_next = torch.stack(batch_next).to(device)

    # ---- Train Discriminator ----
    d_loss = 0.0
    for param in discriminator.parameters():
        param.requires_grad = True  # Unfreeze discriminator for its update step

    batch_size = batch_curr.size(0)
    fake_data, prior_loss = model(
        batch_prev, batch_curr, batch_next, 
        mask_vocal = mask_vocal, mask_inst = mask_inst)
    
    real_labels = torch.ones(batch_size, 1).to(batch_curr.device)
    fake_labels = torch.zeros(batch_size, 1).to(batch_curr.device)
    
    optimizer_D.zero_grad()

    real_preds, inter_data_real = discriminator(batch_curr)
    fake_preds, inter_data_fake = discriminator(fake_data.detach())  # Detach fake data from generator

    real_loss = criterion(real_preds, real_labels)
    fake_loss = criterion(fake_preds, fake_labels)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # ---- Train Generator ----
    g_loss = 0.0
    for param in discriminator.parameters():
        param.requires_grad = False  # Freeze discriminator for its update step

    # Generator tries to fool the discriminator (maximize discriminator's real prediction for fake data)
    fake_preds, inter_data_fake = discriminator(fake_data)
    real_preds, inter_data_real = discriminator(batch_curr)

    g_loss = criterion(fake_preds, real_labels)  # Goal: discriminator should classify fake data as real
    error = F.l1_loss(fake_data, batch_curr, reduction='mean')
    feature_loss = F.l1_loss(inter_data_fake, inter_data_real, reduce="mean")
    final_loss = g_loss * 0.3 + error * 0.5 + feature_loss * 0.1 + prior_loss * 0.1
    
    optimizer.zero_grad()
    final_loss.backward()
    optimizer.step()

    scheduler_D.step()
    scheduler.step()

    print(f"[Epoch {epoch:04d}] | Dis Loss: {d_loss:.4f} | Gen Loss: {g_loss:.4f} | Recon Error: {error:.4f}" )

    if epoch % 200 == 0 or epoch == EPOCHS:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "discriminator_state_dict":discriminator.state_dict()
        }, CHECKPOINT_PATH)
        print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")
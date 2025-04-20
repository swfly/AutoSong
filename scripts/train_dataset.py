#!/usr/bin/env python3

# Curriculum Learning: starting from small size segments 
# and gradually increase to full length

import math
import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

from models.text_encoder     import TextEncoder
from models.audio_encoder    import AudioEncoder
from models.sound_transformer import SoundTransformer


# ──────────────────── helper: load one song ──────────────────── #
DATASET_DIR   = "dataset"
def get_segment_tokens(step, total_steps=10_000, min_len=200, max_len=18_000):
    return 8192
    """Smooth nonlinear segment schedule from 200 to 18k over 10k steps."""
    # Logistic growth curve parameters
    s0 = 4000           # inflection point (shift right if you want slower growth)
    k = 0.0015          # steepness (larger = faster ramp-up)

    # Logistic function: output ∈ (0, 1)
    growth = 1 / (1 + math.exp(-k * (step - s0)))

    # Scaled segment length
    length = min_len + (max_len - min_len) * growth
    return int(length)
def load_random_song(segment_tokens: int):
    """Return (lyrics_embed, inp, tgt) all on GPU, len == MAX_TOKENS."""
    song = random.choice([d for d in os.listdir(DATASET_DIR)
                          if os.path.isdir(os.path.join(DATASET_DIR, d))])
    path = os.path.join(DATASET_DIR, song)

    # --- lyrics ------------------------------------------------- #
    txt = next(f for f in os.listdir(path) if f.endswith(".txt"))
    with open(os.path.join(path, txt), encoding="utf-8") as f:
        lyrics = f.read()
    with torch.no_grad():
        lyr_emb = text_encoder.encode(lyrics).to(device)

    # --- audio → tokens ---------------------------------------- #
    audio = next(f for f in os.listdir(path) if f.endswith((".mp3", ".flac")))
    tok = audio_encoder.encode(os.path.join(path, audio))          # (T, C) on CPU

    if len(tok) > segment_tokens:
        start = random.randint(0, len(tok) - segment_tokens - 1)
        tok = tok[start:start + segment_tokens + 1]
        

    inp = tok.unsqueeze(0).to(device)  # (1, 18k)

    return lyr_emb.detach(), inp.detach()

# ───────────────────────── config ───────────────────────── #

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
CHECKPOINT_PATH = "checkpoints/train_dataset.pt"
DEVICE = device

text_encoder = TextEncoder().to(torch.device("cpu") if DEVICE.type == "mps" else DEVICE)
audio_encoder = AudioEncoder(device="cpu")
VOCAB_PER_CB = audio_encoder.vocab_size
EMBED_DIM = 512
MAX_TOKENS = 18000
EPOCHS = 1000
LR = 1e-4

tokens2d = audio_encoder.encode("dataset/song_001/song_001.mp3")  # (T, C)
N_CODEBOOKS = tokens2d.shape[1]

transformer = SoundTransformer(
    vocab_size=VOCAB_PER_CB,
    n_codebooks=N_CODEBOOKS,
    embed_dim=EMBED_DIM,
    num_heads=2,
    num_layers=3,
    max_seq_len=MAX_TOKENS
).to(DEVICE)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=LR)


if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Found checkpoint at {CHECKPOINT_PATH}, resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["model_state_dict"])
    else:
        transformer.load_state_dict(checkpoint)  # raw state_dict

    if "optimizer_state_dict" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("⚠️ Checkpoint doesn't contain optimizer/scheduler state. Starting fresh for them.")
else:
    print("🆕 No checkpoint found, starting from scratch.")

print("Starting training …")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    transformer.train()
    optimizer.zero_grad()
    segment_tokens = get_segment_tokens(epoch)
    lyr, x = load_random_song(segment_tokens)
    
    logits = transformer(lyr, x)
    # Initialize total loss
    total_loss = 0
    # Loop over each channel (C) to calculate the loss separately per channel
    for c in range(N_CODEBOOKS):
        # Get the logits for the current channel: [B, S, vocab_size]
        channel_logits = logits[:, :, c, :]
        channel_target = x[:, 1:, c]
        channel_logits_flat = channel_logits[:, :-1, :].view(-1, VOCAB_PER_CB)  # [B * (S-1), vocab_size]
        channel_target_flat = channel_target.view(-1)  # [B * (S-1)]
        # Calculate the loss for this channel, compare to NEXT token
        channel_loss = criterion(channel_logits_flat, channel_target_flat)

        # Add it to the total loss
        total_loss += channel_loss
    total_loss /= N_CODEBOOKS
    total_loss.backward()
    optimizer.step()


    print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}")
    del lyr, x, logits, total_loss, channel_logits, channel_target,channel_loss
    torch.cuda.empty_cache()
    gc.collect()

    if epoch % 20 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"💾  saved checkpoint → {CHECKPOINT_PATH}")

quit()
# ───────────────────────── training ───────────────────────── #
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
print(f"🚀 full‑context training (18 k tokens) – device: {device}")

start_epoch = 1

if os.path.exists(CKPT_PATH):
    print(f"🔁 Found checkpoint at {CKPT_PATH}, resuming training...")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # raw state_dict

    if "optimizer_state_dict" in checkpoint and "epoch" in checkpoint:
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("⚠️ Checkpoint doesn't contain optimizer/scheduler state. Starting fresh for them.")
else:
    print("🆕 No checkpoint found, starting from scratch.")

for epoch in range(1, EPOCHS + 1):
    model.train()
    segment_tokens = get_segment_tokens(epoch)
    print('now training with length',segment_tokens)
    lyr, x, y = load_random_song(segment_tokens)

    optimizer.zero_grad(set_to_none=True)
    logits = model(lyr, x)
    loss = criterion(logits.view(-1, TOTAL_VOCAB), y.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()

    # scalar before deleting tensors
    loss_val = loss.item()
    lr_val   = scheduler.get_last_lr()[0]

    # explicit cleanup
    del lyr, x, y, logits, loss
    torch.cuda.empty_cache()
    gc.collect()

    print(f"[{(start_epoch+epoch):03d}/{EPOCHS}] loss={loss_val:.4f}  lr={lr_val:.6f}")

    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, CKPT_PATH)
        print(f"💾  saved checkpoint → {CKPT_PATH}")

#!/usr/bin/env python3
"""
Constant‑VRAM training that still feeds the transformer the full
18 k‑token sequence each step.
"""

import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

from models.text_encoder     import TextEncoder
from models.audio_encoder    import AudioEncoder
from models.sound_transformer import SoundTransformer


# ───────────────────────── config ───────────────────────── #
DATASET_DIR   = "dataset"
VOCAB_PER_CB  = 1024
EMBED_DIM     = 512
MAX_TOKENS    = 18_000          # feed all of them
EPOCHS        = 1000
LR_MAX        = 1e-4
LR_MIN        = 1e-5
CKPT_PATH     = "checkpoints/random_train.pt"

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

# ──────────────────────── encoders ──────────────────────── #
text_enc  = TextEncoder().to("cpu" if device.type == "mps" else device)
audio_enc = AudioEncoder(device="cpu")                # ← stays on CPU

# discover number of EnCodec code‑books
probe_dir   = next(d for d in os.listdir(DATASET_DIR)
                   if os.path.isdir(os.path.join(DATASET_DIR, d)))
probe_audio = next(f for f in os.listdir(os.path.join(DATASET_DIR, probe_dir))
                   if f.endswith((".mp3", ".flac")))
n_codebooks = audio_enc.encode(os.path.join(DATASET_DIR, probe_dir, probe_audio)).shape[1]
TOTAL_VOCAB = n_codebooks * VOCAB_PER_CB

# ─────────────────────── transformer ─────────────────────── #
model = SoundTransformer(
    vocab_size_per_cb = VOCAB_PER_CB,
    n_codebooks       = n_codebooks,
    embed_dim         = EMBED_DIM,
    num_heads         = 4,
    num_layers        = 4,
    max_seq_len       = MAX_TOKENS
).to(device)

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=LR_MAX)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS, eta_min=LR_MIN)


# ──────────────────── helper: load one song ──────────────────── #
def load_random_song():
    """Return (lyrics_embed, inp, tgt) all on GPU, len == MAX_TOKENS."""
    song = random.choice([d for d in os.listdir(DATASET_DIR)
                          if os.path.isdir(os.path.join(DATASET_DIR, d))])
    path = os.path.join(DATASET_DIR, song)

    # --- lyrics ------------------------------------------------- #
    txt = next(f for f in os.listdir(path) if f.endswith(".txt"))
    with open(os.path.join(path, txt), encoding="utf-8") as f:
        lyrics = f.read()
    with torch.no_grad():
        lyr_emb = text_enc.encode(lyrics).to(device)

    # --- audio → tokens ---------------------------------------- #
    audio = next(f for f in os.listdir(path) if f.endswith((".mp3", ".flac")))
    tok2d = audio_enc.encode(os.path.join(path, audio))          # (T, C) on CPU

    offs  = torch.arange(n_codebooks) * VOCAB_PER_CB
    tok   = (tok2d + offs).flatten()[:MAX_TOKENS + 1]            # up to 18 001

    # zero‑pad if shorter (rare)
    if len(tok) < MAX_TOKENS + 1:
        pad = (MAX_TOKENS + 1) - len(tok)
        tok = torch.cat([tok, tok.new_full((pad,), tok[0])])

    inp = tok[:-1].unsqueeze(0).to(device)  # (1, 18k)
    tgt = tok[1: ].unsqueeze(0).to(device)

    return lyr_emb, inp, tgt


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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("⚠️ Checkpoint doesn't contain optimizer/scheduler state. Starting fresh for them.")
else:
    print("🆕 No checkpoint found, starting from scratch.")

for epoch in range(1, EPOCHS + 1):
    model.train()
    lyr, x, y = load_random_song()

    optimizer.zero_grad(set_to_none=True)
    logits = model(lyr, x)
    loss   = criterion(logits.view(-1, TOTAL_VOCAB), y.view(-1))
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

    print(f"[{epoch:03d}/{EPOCHS}] loss={loss_val:.4f}  lr={lr_val:.6f}")

    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, CKPT_PATH)
        print(f"💾  saved checkpoint → {CKPT_PATH}")

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vq_vae import SegmentVQVAE, chunk_encodec_tokens
from models.audio_encoder import AudioEncoder

import torch
import random
import math
from torch.utils.data import DataLoader, TensorDataset



CHECKPOINT_PATH = "checkpoints/vqvae_overfit.pt"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
# ─────────────────────────── config ───────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
SEG_LEN = 256  # EnCodec time-steps per segment (~1.5 seconds)

# ─────────────────────────── data prep ────────────────────────
encoder = AudioEncoder(device=device, bandwidth = 6)
tokens = encoder.encode("dataset/song_001/song_001.mp3")  # (T, C)
tokens = tokens.to(device)

segments = chunk_encodec_tokens(tokens, seg_len=SEG_LEN)  # list of (S, C)

if len(segments) < 3:
    raise ValueError("Song too short to form [prev | curr | next] segments.")

# Just try one triplet for test
tokens_prev = segments[0].unsqueeze(0)  # [1, T_seg, C]
tokens_curr = segments[1].unsqueeze(0)
tokens_next = segments[2].unsqueeze(0)


# ─────────────────────────── model init ───────────────────────
V = encoder.vocab_size
C = tokens_prev.shape[-1]

model = SegmentVQVAE(
    vocab_size=V,
    n_codebooks=C,
    seg_len=SEG_LEN,
    latent_dim=128,
    num_codes=128,
    beta=0.5
).to(device)


# ─────────────────────────── training setup ───────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
EPOCHS = 5000
zero_second_prob = 0.0  # set to >0.0 if you want to experiment with partial mute
start_epoch = 1
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"⏩ Resuming from epoch {start_epoch}")

# ─────────────────────────── training loop ────────────────────────────

# Create triplets in batch
tokens_prev = torch.stack(segments[:-2]).to(device)  # [B, T, C]
tokens_curr = torch.stack(segments[1:-1]).to(device)
tokens_next = torch.stack(segments[2:]).to(device)
tokens_prev = torch.stack(segments[0:1]).to(device)  # [B, T, C]
tokens_curr = torch.stack(segments[1:2]).to(device)
tokens_next = torch.stack(segments[2:3]).to(device)
triplet_dataset = TensorDataset(tokens_prev, tokens_curr, tokens_next)
model.train()
print("🚀 Starting overfitting loop on single song …")
BATCH_SIZE = 1
loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
div_loss_scale = 0.0
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    total_loss = total_recon = total_vq = total_div = 0.0

    for batch in loader:
        batch_prev, batch_curr, batch_next = [x.to(device) for x in batch]

        optimizer.zero_grad()
        loss, metrics = model(
            batch_prev, batch_curr, batch_next,
            zero_second_prob=zero_second_prob,
            variance_loss_scale=div_loss_scale
        )
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_recon   += metrics["recon_loss"].item()
        total_vq      += metrics["vq_loss"].item()
        total_div     += metrics["diversity_loss"].item()


    # Average metrics per epoch
    num_batches = len(loader)
    avg_loss  = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_vq    = total_vq / num_batches
    avg_div = total_div / num_batches
    print(f"[Epoch {epoch:03d}] Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | VQ: {avg_vq:.4f} | DIV: {avg_div:.4f}")

    if epoch % 50 == 0 or epoch == EPOCHS:
        # Sample one batch to measure code usage
        with torch.no_grad():
            _, code_ids, _ = model.vq(model.encoder(batch_curr))
            codes_used = code_ids.unique().numel()
            div_loss_scale = 0.01 * (1.0 - math.exp(-0.02 * (codes_used - 1)))

        print(f"🧱 Unique codes used: {codes_used} | div loss scale = {div_loss_scale:.4f}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

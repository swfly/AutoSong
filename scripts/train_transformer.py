# scripts/train_transformer.py   ← FULL UPDATED VERSION
import os, sys, random, math, gc
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.text_encoder               import TextEncoder
from models.sound_transformer import SoundTransformerContinuous
from models.autoencoder                import SegmentAutoEncoder, SpectrogramDiscriminator
from utils.latent_visualizer           import visualize_latents   # (optional)

# ───────────────────────── Config ───────────────────────── #
DATASET_DIR   = "dataset"
VAE_CKPT      = "checkpoints/vqvae_dataset.pt"            # AE + D weights
TRANS_CKPT    = "checkpoints/train_latent_transformer.pt"
LATENT_FILE   = "cached_latents.pt"

LATENT_C, H, W = 4, 32, 32
LATENT_D       = H * W
MAX_SEQ_LEN    = 256
BATCH_SIZE     = 2
EPOCHS         = 500_000

λ_FEAT = 0.10      # feature-matching weight
λ_ADV  = 0.10      # adversarial weight

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ───────── Text Encoder ───────── #
text_enc = TextEncoder(max_tokens=512).to("cpu")

# ───────── Load frozen AE & Discriminator ───────── #
ae   = SegmentAutoEncoder(
        input_dim=256, latent_size=(H, W), latent_channels=LATENT_C,
        network_channel_base=32, seq_len=256
      ).to(device).eval()

disc = SpectrogramDiscriminator(1, base_dim=12).to(device).eval()

vae_state = torch.load(VAE_CKPT, map_location=device)
ae.load_state_dict(vae_state["model_state_dict"], strict=False)
disc.load_state_dict(vae_state["discriminator_state_dict"], strict=False)

for p in ae.parameters():   p.requires_grad = False
for p in disc.parameters(): p.requires_grad = False

# ───────── Transformer ───────── #
txf = SoundTransformerContinuous(
    in_channels=LATENT_C,
    patch_hw=(H,W),
    d_model=1024, 
    n_heads=16, 
    n_layers=16,
    max_seq_len=MAX_SEQ_LEN
    ).to(device)


LR = 2e-4
def build_optimizer_and_scheduler(model, base_lr=1e-4, warmup_steps=1000, total_steps=500_000):
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)
    # Warmup scheduler
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    return optimizer, scheduler
opt, sched = build_optimizer_and_scheduler(
    txf, base_lr = LR, warmup_steps=32, total_steps=EPOCHS)

start_epoch = 1
if os.path.exists(TRANS_CKPT):
    ck = torch.load(TRANS_CKPT, map_location=device)
    fp16_state_dict = ck["model_state_dict"]
    fp32_state_dict = {k: v.float() for k, v in fp16_state_dict.items()}
    txf.load_state_dict(fp32_state_dict)
    # opt.load_state_dict(ck["optimizer_state_dict"])   # enable if you wish
    start_epoch = 1
    print(f"🔁 Resumed from epoch {start_epoch}")

# ───────── Data helpers ───────── #

song_list = [d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))]
def load_random_song():
    song = random.choice(song_list)
    path = os.path.join(DATASET_DIR, song)
    lat_path = os.path.join(path, LATENT_FILE)

    try:
        lat_list = torch.load(lat_path, map_location="cpu")  # list of [1,C,H,W]
    except Exception:
        return None, None

    if len(lat_list) < 2:
        return None, None

    # lyrics
    try:
        lyr_file = next(f for f in os.listdir(path) if f.endswith(".txt"))
        with open(os.path.join(path, lyr_file), encoding="utf-8") as f:
            lyr_txt = f.read()
    except Exception:
        lyr_txt = "..."

    lyr_emb = text_enc.encode(lyr_txt).to(device)           # [1,T_text,D]

    lat = torch.stack(lat_list, dim=0)  # [S,C,H,W]
    return lyr_emb, lat

# When loading in batch, truncates the longer tracks to the shortest size
def load_batch(bs=BATCH_SIZE):
    lyr_batch, lat_batch = [], []
    while len(lat_batch) < bs:
        lyr, lat = load_random_song()
        if lyr is None: continue
        lyr_batch.append(lyr)
        lat_batch.append(lat)
    min_len = min(z.size(0) for z in lat_batch)
    max_len = min(MAX_SEQ_LEN, min_len)
    lyr_out = torch.cat(lyr_batch, dim=0)                       # [B,T_text,D]
    lat_out = torch.stack([z[:max_len] for z in lat_batch])     # [B,S,C,D]
    return lyr_out, lat_out.to(device)

# ───────── Training Loop ───────── #
print("🚀 Starting training …")
os.makedirs(os.path.dirname(TRANS_CKPT), exist_ok=True)

# min_len = 256
# max_len = MAX_SEQ_LEN
# step = 60000000000     # how often to grow
training_sequence_length = MAX_SEQ_LEN
train_losses = []
for epoch in range(start_epoch, EPOCHS + 1):
    txf.train()
    opt.zero_grad(set_to_none=True)

    lyrics, z = load_batch()
    # training_sequence_length = min(z.shape[1], max_len, min_len * (2 ** (epoch // step)))
    T = z.shape[1]
    if T > training_sequence_length:
        start = torch.randint(0, T - training_sequence_length + 1, (1,)).item()
    else:
        start = 0  # if sequence is already shorter than training window
    z = z[:,start:start + training_sequence_length]
    z_in  = z
    z_tgt = z

    pred = txf(lyrics, z_in)                     # [B,S-1,C,D]

    # -------- latent L1 --------
    loss_lat = nn.functional.l1_loss(pred[:, :-1] - z[:, :-1], z[:, 1:] - z[:,:-1], reduction="mean")

    # # -------- perceptual & adversarial --------
    # pred, z_tgt : [B, S, C, D]   (with S ≥ 3)
    B, S, C, H, W = pred.shape
    half = C // 2          # 2 channels per split
    
    rand_idx = torch.randint(low=1, high=S-1, size=(B,))
    triplets_pred = []
    triplets_real = []

    for b in range(B):
        i = rand_idx[b]
        triplets_pred.append(torch.cat([
            pred[b, i - 1], pred[b, i], pred[b, i + 1]
        ], dim=0))  # [3*C, D]

        triplets_real.append(torch.cat([
            z_tgt[b, i - 1], z_tgt[b, i], z_tgt[b, i + 1]
        ], dim=0))  # [3*C, D]

    pred_flat = torch.stack(triplets_pred).view(B, 3*C, H, W)  # [B, 3C, H, W]
    tgt_flat  = torch.stack(triplets_real).view(B, 3*C, H, W)

    # ----- decode to mel -----
    mel_pred = ae.decoder(pred_flat)              # [(B*(S-2)), seg_len, mel_bins]
    mel_real = ae.decoder(tgt_flat)

    # ----- discriminator features & verdict -----
    disc_fake, feat_fake = disc(mel_pred)
    disc_real, feat_real = disc(mel_real)

    # feature-matching loss
    loss_feat = nn.functional.l1_loss(feat_fake, feat_real, reduction="mean")

    # adversarial loss (generator wants 'real' label)
    real_lbl = torch.ones_like(disc_fake)
    loss_adv = nn.functional.binary_cross_entropy(disc_fake, real_lbl)

    # -------- total loss --------
    # loss = loss_lat + λ_FEAT * loss_feat + λ_ADV * loss_adv
    loss = loss_lat + λ_FEAT * loss_feat
    # loss = loss_lat
    loss.backward()
    opt.step()
    sched.step()
    train_losses.append(loss.item())
    # print(f"[{epoch:04d}] L1 {loss_lat:.4f}")
    print(f"[{epoch:04d}] L1 {loss_lat:.4f} | Feat {loss_feat:.4f} | Adv {loss_adv:.4f}")

    # --- optional live viz every 1000 steps ---
    # if epoch % 1 == 0:
    #     visualize_latents(
    #         [pred[0,1].view(1,C,H,W).cpu(), z_tgt[0,1].view(1,C,H,W).cpu()],
    #         ["pred","target"]
    #     )

    # --- checkpoint ---
    if epoch % 1000 == 0:
        fp16_state_dict = {k: v.half() for k, v in txf.state_dict().items()}
        torch.save({
            "epoch": epoch,
            "model_state_dict": fp16_state_dict,
            "train_losses": train_losses
        }, TRANS_CKPT)
        print("💾 saved", TRANS_CKPT)

# scripts/sample_latent_transformer.py
import os, sys, torch, torchaudio, math
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.text_encoder          import TextEncoder
from models.autoencoder           import SegmentAutoEncoder     # your AE
from models.sound_transformer import SoundTransformerContinuous
from utils.chunk_segments         import chunk_segments
from tqdm import tqdm

# ───────────── paths & constants ─────────────
DATASET_DIR       = "dataset"
TRANS_CKPT        = "checkpoints/train_latent_transformer.pt"
AE_CKPT           = "checkpoints/vqvae_dataset.pt"     # same ckpt used for caching
LATENT_FILE       = "cached_latents.pt"                # generated earlier
OUT_WAV           = "predicted.wav"

LATENT_C     = 4           # 6 blocks × 2 splits
H, W     =  (32, 32)      # flattened H×W
SEG_LEN      = 256          # mel frames per segment
SEQ          = 32          # number of *new* segments to sample
PREFIX_LEN   = 4           # seed length from dataset
EMBED_DIM    = 512
AUTOREGRESSION_LEN = 256

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ───────────── models ─────────────
print("🔧 loading models …")
text_enc = TextEncoder(max_tokens=512).to("cpu")

ae = SegmentAutoEncoder(
    input_dim=256, latent_size=(32,32), latent_channels=4,
    network_channel_base=32, seq_len=SEG_LEN
).to(device).eval()
ae.load_state_dict(torch.load(AE_CKPT, map_location=device)["model_state_dict"])

txf = SoundTransformerContinuous(
    in_channels=LATENT_C,
    patch_hw=(H,W),
    d_model=1024, 
    n_heads=16, 
    n_layers=16,
    max_seq_len=AUTOREGRESSION_LEN
    ).to(device).eval()

ck = torch.load(TRANS_CKPT, map_location=device)
fp16_state_dict = ck["model_state_dict"]
fp32_state_dict = {k: v.float() for k, v in fp16_state_dict.items()}
txf.load_state_dict(fp32_state_dict)

# ───────────── pick a prefix sample ─────────────
song = sorted([d for d in os.listdir(DATASET_DIR) if d.startswith("song_")])[0]
lat_path = os.path.join(DATASET_DIR, song, LATENT_FILE)
lat_list = torch.load(lat_path, map_location=device)              # list of [1,C,H,W]

# initial sequence
generated = lat_list[:PREFIX_LEN]

# ───────────── lyrics conditioning ─────────────
with open("test_song.txt", encoding="utf-8") as f:
    lyrics_txt = f.read()
lyr_ids = text_enc.encode(lyrics_txt).to(device)

# ───────────── autoregressive sampling ─────────────
print("🎼 generating …")
with torch.inference_mode():

    for step in tqdm(range(SEQ)):
        inp = torch.stack(generated, dim=0).unsqueeze(0)     # [1, S, C, D]
        pred = txf(lyr_ids, inp.to(device))                # [1,S,C,D]
        next_lat = pred[:, -1]                  # last step  [1,C,D]
        generated.append(next_lat.squeeze(0))

    lat_seq = torch.stack(generated, dim=0)                   # [(P+SEQ),C,D]
    lat_seq = lat_seq.view(-1, LATENT_C, 32, 32)              # reshape back

    # ───────────── decode to mel & wav ─────────────
    print("🔊 decoding …")
    mels = []
    for i in range(1, lat_seq.size(0)-1):                     # skip edges
        z_prev, z_curr, z_next = lat_seq[i-1], lat_seq[i], lat_seq[i+1]
        z_triplet = torch.cat([                                # match AE decoder input
            z_prev,
            z_curr,
            z_next
        ], dim=0).unsqueeze(0).to(device)                     # [1,6C/2,H,W]
        mel = ae.decoder(z_triplet).cpu()                     # (1, SEG_LEN, 256)
        mels.append(mel.squeeze(0))

    full_mel = torch.cat(mels, dim=0)                         # [T_total, 256]

    # Griffin–Lim back to waveform
    from models.audio_encoder import AudioEncoder
    aud_enc = AudioEncoder(device="cpu")
    wave = aud_enc.decode(full_mel)

torchaudio.save(OUT_WAV, wave.detach(), aud_enc.sample_rate)
print(f"✅  saved {Path(OUT_WAV).resolve()}")

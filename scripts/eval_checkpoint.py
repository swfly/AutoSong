#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import torch, torchaudio

from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.sound_transformer import SoundTransformer

WINDOW = 4096
WAV_PATH = "dataset/song_001/song_001.mp3"
CKPT = "checkpoints/random_train.pt"
VOCAB_PER_CB, EMB = 1024, 512
MAX_TOKENS = 8192

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"🔌  device: {device}   │  sliding window: {WINDOW}")

txt_enc = TextEncoder(model_name="hfl/chinese-roberta-wwm-ext").to(torch.device("cpu") if device.type == "mps" else device)
aud_enc = AudioEncoder(device=device)

tokens2d = aud_enc.encode(WAV_PATH)
N_CODEBOOKS = tokens2d.shape[1]
offset = torch.arange(N_CODEBOOKS) * VOCAB_PER_CB
ref = (tokens2d + offset).flatten()
TOTAL_VOCAB = VOCAB_PER_CB * N_CODEBOOKS

print(f"📦  {N_CODEBOOKS} EnCodec codebooks detected")

model = SoundTransformer(VOCAB_PER_CB, N_CODEBOOKS, EMB,
                         num_heads=4, num_layers=3,
                         max_seq_len=MAX_TOKENS).to(device).eval()
if os.path.exists(CKPT):
    print(f"🔁 Found checkpoint at {CKPT}, load weights...")
    checkpoint = torch.load(CKPT, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # raw state_dict

with open("dataset/song_001/song_001.txt", encoding="utf-8") as f:
    lyrics = f.read()
with torch.no_grad():
    lyr_emb = txt_enc.encode(lyrics).to(device)

BOS, SEQ = ref[0].item(), min(len(ref), MAX_TOKENS)
SEQ = 1000
print(f"🎯  generating {SEQ} tokens (incl. BOS)")

known_tokens = ref[:128]  # Change this to any number of tokens you'd like to start with
gen = known_tokens.unsqueeze(0).to(device)

# gen = torch.tensor([[BOS]], device=device)
past_kv = None
pos_idx = 128
cache_len = 128

print("🎼  generating …")
with torch.inference_mode():
    for _ in range(SEQ - 1):
        print("predicting", _)
        logits = model(lyr_emb, gen[:,:pos_idx])
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, nxt], 1)
        pos_idx += 1
        cache_len += 1

tokens = gen.squeeze(0).cpu()
rem = tokens.numel() % N_CODEBOOKS
if rem: tokens = tokens[:-rem]

tokens2d = (tokens.view(-1, N_CODEBOOKS)) % VOCAB_PER_CB  # remove offset
print("🔊  decoding …")
wave = aud_enc.decode(tokens2d)
sr = aud_enc.sample_rate

out = Path("create.wav")
torchaudio.save(out.as_posix(), wave, sr)
print(f"✅  saved {out.resolve()}")

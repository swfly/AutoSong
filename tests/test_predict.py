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
LYRICS = "hello boy"
CKPT = "checkpoints/overfit_song.pt"
VOCAB_PER_CB, EMB = 1024, 512
MAX_TOKENS = 18_000

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"🔌  device: {device}   │  sliding window: {WINDOW}")

txt_enc = TextEncoder().to(torch.device("cpu") if device.type == "mps" else device)
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
model.load_state_dict(torch.load(CKPT, map_location=device), strict=True)

with torch.no_grad():
    lyr_emb = txt_enc.encode(LYRICS).to(device)

BOS, SEQ = ref[0].item(), min(len(ref), MAX_TOKENS)
print(f"🎯  generating {SEQ} tokens (incl. BOS)")

gen = torch.tensor([[BOS]], device=device)
past_kv = None
pos_idx = 0
cache_len = 0

print("🎼  generating …")
with torch.inference_mode():
    for _ in range(SEQ - 1):
        print("predicting",_)
        logits, past_kv = model(
            lyr_emb,
            gen[:, -1:],
            past_kv=past_kv,
            use_cache=True,
            step=pos_idx
        )
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, nxt], 1)
        pos_idx += 1
        cache_len += 1
        if cache_len > WINDOW:
            cache_len = WINDOW
            for i, (k, v) in enumerate(past_kv):
                past_kv[i] = (k[:, -WINDOW:], v[:, -WINDOW:])

tokens = gen.squeeze(0).cpu()
rem = tokens.numel() % N_CODEBOOKS
if rem: tokens = tokens[:-rem]

tokens2d = (tokens.view(-1, N_CODEBOOKS)) % VOCAB_PER_CB  # remove offset
print("🔊  decoding …")
wave = aud_enc.decode(tokens2d)
sr = aud_enc.sample_rate

out = Path("reconstructed2.wav")
torchaudio.save(out.as_posix(), wave, sr)
print(f"✅  saved {out.resolve()}")

acc = (tokens[:SEQ] == ref[:SEQ]).float().mean()
print(f"📊  accuracy vs reference: {acc * 100:.2f}%")

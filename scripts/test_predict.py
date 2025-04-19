#!/usr/bin/env python3
"""
Generate the over‑fitted song using a fixed‑size sliding KV cache.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import torch, torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.sound_transformer import SoundTransformer


# ─────────────── settings ─────────────── #
WINDOW      = 2048                 # keep this many past tokens per layer
WAV_PATH    = "test.mp3"
LYRICS      = "I saw the sun rise over the hills and I felt peace."
CKPT        = "checkpoints/overfit_song.pt"
VOCAB, EMB  = 1024, 512
MAX_TOKENS  = 18_000

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"🔌  device: {device}   │  sliding window: {WINDOW}")


# ─────────────── initialise ─────────────── #
txt_enc  = TextEncoder().to(torch.device("cpu") if device.type == "mps" else device)
aud_enc  = AudioEncoder(device=device)

N_CODEBOOKS = aud_enc.encode(WAV_PATH).shape[1]
print(f"📦  {N_CODEBOOKS} EnCodec code‑books detected")

model = SoundTransformer(VOCAB, EMB, num_heads=4, num_layers=3,
                         max_seq_len=MAX_TOKENS).to(device).eval()
model.load_state_dict(torch.load(CKPT, map_location=device), strict=True)


# ─────────────── conditioning ─────────────── #
with torch.no_grad():
    lyr_emb = txt_enc.encode(LYRICS).to(device)

ref = aud_enc.encode(WAV_PATH).flatten()
BOS, SEQ = ref[0].item(), min(len(ref), MAX_TOKENS)
print(f"🎯  generating {SEQ} tokens (incl. BOS)")


# ─────────────── generation ─────────────── #
gen     = torch.tensor([[BOS]], device=device)
past_kv = None
step    = 0

print("🎼  generating …")
with torch.inference_mode():
    for _ in range(SEQ - 1):
        print("predicting",_)
        logits, past_kv = model(
            lyr_emb, gen[:, -1:], past_kv=past_kv,
            use_cache=True, step=step
        )
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        gen = torch.cat([gen, nxt], 1)
        step += 1

        # bound the cache size
        if step > WINDOW:
            step = WINDOW
            for i, (k, v) in enumerate(past_kv):
                past_kv[i] = (k[:, -WINDOW:], v[:, -WINDOW:])

tokens = gen.squeeze(0).cpu()

rem = tokens.numel() % N_CODEBOOKS
if rem: tokens = tokens[:-rem]
tokens2d = tokens.view(-1, N_CODEBOOKS)


# ─────────────── decode & save ─────────────── #
print("🔊  decoding …")
wave = aud_enc.decode(tokens2d)
sr   = aud_enc.sample_rate

out = Path("reconstructed.wav")
torchaudio.save(out.as_posix(), wave, sr)
print(f"✅  saved {out.resolve()}")

acc = (tokens[:SEQ] == ref[:SEQ]).float().mean()
print(f"📊  accuracy vs reference: {acc * 100:.2f}%")

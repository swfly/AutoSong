
import math
import os, sys, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import torch, torchaudio, torch.nn as nn, torch.optim as optim

from models.text_encoder     import TextEncoder
from models.audio_encoder    import AudioEncoder
from models.sound_transformer import SoundTransformer


# ───────────────────────── config ───────────────────────── #

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
CHECKPOINT_PATH = "checkpoints/train_dataset.pt"
DEVICE = device

text_encoder = TextEncoder(max_tokens=512).to(torch.device("cpu"))
audio_encoder = AudioEncoder(device="cpu")
VOCAB_PER_CB = audio_encoder.vocab_size
EMBED_DIM = 1024
MAX_TOKENS = 18000
EPOCHS = 1000
LR = 5e-5
tokens2d = audio_encoder.encode("dataset/song_001/song_001.mp3")  # (T, C)
N_CODEBOOKS = tokens2d.shape[1]

transformer = SoundTransformer(
    vocab_size=VOCAB_PER_CB,
    n_codebooks=N_CODEBOOKS,
    embed_dim=EMBED_DIM,
    num_heads=4,
    num_layers=8,
    max_seq_len=MAX_TOKENS
).to(DEVICE)



# ───────────────────────── load ───────────────────────── #
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔁 Found checkpoint at {CHECKPOINT_PATH}, resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["model_state_dict"])
    else:
        transformer.load_state_dict(checkpoint)  # raw state_dict
else:
    print("🆕 No checkpoint found.")
    quit()


# ───────────────────────── generate ───────────────────────── #
# lyrics = "明天你好\n在一片蓝色月光下"
with open("test_song.txt", encoding="utf-8") as f:
    lyrics = f.read()
with torch.no_grad():
    lyr_emb = text_encoder.encode(lyrics).to(device)
# 🪄 Use a longer prefix from the reference sequence instead of BOS only
PREFIX_LEN = 32  # or 512, depending on how much context you want
SEQ = 1024
prefix = tokens2d[:PREFIX_LEN].unsqueeze(0).to(DEVICE)  # shape [1, P]
gen = prefix.clone()  # will grow during generation
pos_idx = PREFIX_LEN  # ⬅️ start from here

print("🎼  generating …")
with torch.inference_mode():
    for _ in range(SEQ):
        print("predicting", _)
        logits = transformer(lyr_emb, gen)
        last_token_logits = logits[:, -1]
        # For each channel, get the next token by selecting the argmax across vocab_size
        next_tokens = []
        for c in range(N_CODEBOOKS):  # Loop over channels (codebooks)
            next_token_c = last_token_logits[:, c, :].argmax(-1, keepdim=True)  # Shape: [B, 1]
            next_tokens.append(next_token_c)
        
        nxt = torch.cat(next_tokens, dim=1).unsqueeze(0)
        gen = torch.cat([gen, nxt], 1)
        pos_idx += 1

tokens = gen.squeeze(0).cpu()
print("🔊  decoding …")
wave = audio_encoder.decode(tokens)
sr = audio_encoder.sample_rate

out = Path("predicted.wav")
torchaudio.save(out.as_posix(), wave, sr)
print(f"✅  saved {out.resolve()}")

acc = (tokens[:PREFIX_LEN + SEQ] == tokens2d[:PREFIX_LEN + SEQ]).float().mean()
print(f"📊  accuracy vs reference: {acc * 100:.2f}%")
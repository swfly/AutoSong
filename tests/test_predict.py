#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import torch, torchaudio

from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.sound_transformer import SoundTransformer

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
WAV_PATH = "dataset/song_001/song_001.mp3"
LYRICS_TEXT = """I saw the sun rise over the hills and I felt peace."""
CHECKPOINT_PATH = "checkpoints/overfit_song.pt"
DEVICE = get_best_device()
text_encoder = TextEncoder().to(torch.device("cpu") if DEVICE.type == "mps" else DEVICE)
audio_encoder = AudioEncoder(device=DEVICE)
VOCAB_PER_CB = audio_encoder.vocab_size
EMBED_DIM = 512
MAX_TOKENS = 18000
EPOCHS = 300
LR = 1e-3

print("Encoding lyrics …")
lyrics_embed = text_encoder.encode(LYRICS_TEXT).to(DEVICE)

print("Encoding audio …")
tokens2d = audio_encoder.encode(WAV_PATH)  # (T, C)
N_CODEBOOKS = tokens2d.shape[1]
print(f"🔌  device: {DEVICE}")

transformer = SoundTransformer(
    vocab_size=VOCAB_PER_CB,
    n_codebooks=N_CODEBOOKS,
    embed_dim=EMBED_DIM,
    num_heads=2,
    num_layers=2,
    max_seq_len=MAX_TOKENS
).to(DEVICE)
transformer.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=True)
transformer.eval()

# 🪄 Use a longer prefix from the reference sequence instead of BOS only
PREFIX_LEN = 128  # or 512, depending on how much context you want
SEQ = 128
prefix = tokens2d[:PREFIX_LEN].unsqueeze(0).to(DEVICE)  # shape [1, P]
gen = prefix.clone()  # will grow during generation
pos_idx = PREFIX_LEN  # ⬅️ start from here


print("🎼  generating …")
with torch.inference_mode():
    for _ in range(SEQ):
        print("predicting", _)
        logits = transformer(lyrics_embed, gen)
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

out = Path("reconstructed2.wav")
torchaudio.save(out.as_posix(), wave, sr)
print(f"✅  saved {out.resolve()}")

acc = (tokens[:PREFIX_LEN + SEQ] == tokens2d[:PREFIX_LEN + SEQ]).float().mean()
print(f"📊  accuracy vs reference: {acc * 100:.2f}%")

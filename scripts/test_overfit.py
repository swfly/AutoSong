import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.text_encoder import TextEncoder
from models.audio_encoder import AudioEncoder
from models.sound_transformer import SoundTransformer  # formerly MusicTransformer

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # for Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")
# -------------------------------
# Config
# -------------------------------
WAV_PATH = "test.mp3"
LYRICS_TEXT = """I saw the sun rise over the hills and I felt peace."""
CHECKPOINT_PATH = "checkpoints/overfit_song.pt"
VOCAB_SIZE = 1024
NUM_CODEBOOKS = 2
EMBED_DIM = 512
MAX_TOKENS = 18000  # depends on length of audio and flattening
DEVICE = get_best_device()
EPOCHS = 300
LR = 1e-3

# -------------------------------
# Prepare Models
# -------------------------------
text_encoder = TextEncoder().to(torch.device("cpu") if DEVICE.type == "mps" else DEVICE)
audio_encoder = AudioEncoder(device=DEVICE)

transformer = SoundTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=4,
    num_layers=3,
    max_seq_len=MAX_TOKENS
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=LR)

# -------------------------------
# Encode Inputs
# -------------------------------
print("Encoding lyrics...")
lyrics_embed = text_encoder.encode(LYRICS_TEXT).to(DEVICE)  # (1, D)

print("Encoding audio...")
tokens = audio_encoder.encode(WAV_PATH)  # (T, C)
tokens = tokens.reshape(-1)              # interleave codebooks: (T * C,)

tokens = tokens[:MAX_TOKENS + 1]         # ensure length
input_tokens = tokens[:-1].unsqueeze(0).to(DEVICE)   # (1, L)
target_tokens = tokens[1:].unsqueeze(0).to(DEVICE)   # (1, L)

# -------------------------------
# Training Loop
# -------------------------------
print("Starting training...")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    transformer.train()
    optimizer.zero_grad()

    logits = transformer(lyrics_embed, input_tokens)  # (1, L, V)
    loss = criterion(logits.view(-1, VOCAB_SIZE), target_tokens.view(-1))

    loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    if epoch % 50 == 0:
        torch.save(transformer.state_dict(), CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

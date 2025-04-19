import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
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

WAV_PATH = "test.mp3"
LYRICS_TEXT = """I saw the sun rise over the hills and I felt peace."""
CHECKPOINT_PATH = "checkpoints/overfit_song.pt"

VOCAB_PER_CB = 1024
EMBED_DIM = 512
MAX_TOKENS = 18000
EPOCHS = 300
LR = 1e-3
DEVICE = get_best_device()

text_encoder = TextEncoder().to(torch.device("cpu") if DEVICE.type == "mps" else DEVICE)
audio_encoder = AudioEncoder(device=DEVICE)

print("Encoding lyrics …")
lyrics_embed = text_encoder.encode(LYRICS_TEXT).to(DEVICE)

print("Encoding audio …")
tokens2d = audio_encoder.encode(WAV_PATH)  # (T, C)
N_CODEBOOKS = tokens2d.shape[1]
TOTAL_VOCAB = VOCAB_PER_CB * N_CODEBOOKS
offset = torch.arange(N_CODEBOOKS) * VOCAB_PER_CB
tokens_offset = (tokens2d + offset).flatten()

tokens = tokens_offset[:MAX_TOKENS + 1]
input_tokens = tokens[:-1].unsqueeze(0).to(DEVICE)
target_tokens = tokens[1:].unsqueeze(0).to(DEVICE)

transformer = SoundTransformer(
    vocab_size_per_cb=VOCAB_PER_CB,
    n_codebooks=N_CODEBOOKS,
    embed_dim=EMBED_DIM,
    num_heads=4,
    num_layers=3,
    max_seq_len=MAX_TOKENS
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=LR)

print("Starting training …")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    transformer.train()
    optimizer.zero_grad()

    logits = transformer(lyrics_embed, input_tokens)
    loss = criterion(logits.view(-1, TOTAL_VOCAB), target_tokens.view(-1))

    loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    if epoch % 50 == 0:
        torch.save(transformer.state_dict(), CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

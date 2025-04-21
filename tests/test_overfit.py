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

transformer = SoundTransformer(
    vocab_size=VOCAB_PER_CB,
    n_codebooks=N_CODEBOOKS,
    embed_dim=EMBED_DIM,
    num_heads=2,
    num_layers=2,
    max_seq_len=MAX_TOKENS
).to(DEVICE)

input_tokens2d = tokens2d.unsqueeze(0).to(DEVICE)

# #####
# transformer.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=True)
# with torch.no_grad():
#     logits = transformer(lyrics_embed, input_tokens2d[:,:5])
#     print(input_tokens2d[:,:5])
#     # print(logits[0, 127])  # raw logits
#     # print(f"Logits shape: {logits.shape}")
#     print(f"Predicted token: {logits[0, 4, :].argmax(-1)}")  # predicted token (argmax)
# quit()
# ####


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr=LR)

print("Starting training …")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    transformer.train()
    optimizer.zero_grad()

    logits = transformer(lyrics_embed, input_tokens2d)  # [B, S, C, V]

    # Predicting next token: logits[:, :-1], targets[:, 1:]
    logits = logits[:, :-1, :, :]    # [B, S-1, C, V]
    targets = input_tokens2d[:, 1:, :]            # [B, S-1, C]

    # Reshape to flatten everything:
    # logits_flat: [B * (S-1) * C, V]
    # targets_flat: [B * (S-1) * C]
    logits_flat = logits.reshape(-1, VOCAB_PER_CB)
    targets_flat = targets.reshape(-1)

    # Compute total cross-entropy loss in parallel
    total_loss = criterion(logits_flat, targets_flat)
    total_loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}")

    if epoch % 50 == 0:
        torch.save(transformer.state_dict(), CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

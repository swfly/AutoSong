import torch
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.sound_transformer import SoundTransformer

# --------------------------
# Config
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAV_PATH = "test.mp3"  # Replace with your test file
LYRICS_TEXT = "I saw the sun rise over the hills and I felt peace."

# Transformer settings (match EnCodec + embedding)
NUM_CODEBOOKS = 8              # EnCodec: 8 quantizers
VOCAB_SIZE = 1024              # EnCodec vocab per codebook
SEQ_LEN = 750                  # Can be adjusted to control context window
EMBED_DIM = 512                # Must match TextEncoder projection

# --------------------------
# Load Models
# --------------------------
text_encoder = TextEncoder().to(DEVICE)
audio_encoder = AudioEncoder(device=DEVICE)
transformer = SoundTransformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=8,
    num_layers=6,
    max_seq_len=SEQ_LEN
).to(DEVICE)

transformer.eval()

# --------------------------
# Encode Inputs
# --------------------------
# 1. Lyrics → Embedding
lyrics_embed = text_encoder.encode(LYRICS_TEXT).to(DEVICE)   # (1, D)

# 2. Audio → Token Sequence
tokens = audio_encoder.encode(WAV_PATH)                      # (T, C)
print(f"Original audio tokens shape: {tokens.shape}")        # e.g., (1024, 8)

# We'll just test on the first codebook for now (C = 0) to simplify
tokens = tokens[:, 0]                                        # (T,)
tokens = tokens[:SEQ_LEN]                                    # truncate for test
tokens = tokens.unsqueeze(0).to(DEVICE)                      # (1, T)

# --------------------------
# Run Forward Pass
# --------------------------
with torch.no_grad():
    logits = transformer(lyrics_embed, tokens)               # (1, T, vocab_size)

print(f"Transformer output shape: {logits.shape}")           # should be (1, T, VOCAB_SIZE)
print("Autoregressive test forward pass succeeded.")

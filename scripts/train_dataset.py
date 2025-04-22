import math
import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch, torch.nn as nn, torch.optim as optim

from models.text_encoder     import TextEncoder
from models.audio_encoder    import AudioEncoder
from models.sound_transformer import SoundTransformer

DATASET_DIR   = "dataset"
def get_segment_tokens(step, total_steps=10_000, min_len=200, max_len=18_000):
    return 4096

def load_random_song(segment_tokens: int):
    song = random.choice([d for d in os.listdir(DATASET_DIR)
                          if os.path.isdir(os.path.join(DATASET_DIR, d))])
    path = os.path.join(DATASET_DIR, song)
    cache_path = os.path.join(path, "cached_encoding.pt")
    print("\U0001F50D loading sample from", path)

    if os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, map_location="cpu")
            lyrics = data["lyrics"]
            tokens = data["tokens"]
            print("✅ loaded cache")
        except Exception as e:
            print("⚠️ failed to load cache:", e)
            lyrics, tokens = None, None
    else:
        lyrics, tokens = None, None

    if lyrics is None or tokens is None:
        print("🛠️  computing new encoding...")
        txt = next(f for f in os.listdir(path) if f.endswith(".txt"))
        with open(os.path.join(path, txt), encoding="utf-8") as f:
            lyrics_txt = f.read()
        garbage = "~" * random.randint(0, 100)
        lyrics_txt = garbage + lyrics_txt
        with torch.no_grad():
            lyrics = text_encoder.encode(lyrics_txt).cpu()

        audio = next(f for f in os.listdir(path) if f.endswith((".mp3", ".flac")))
        tokens = audio_encoder.encode(os.path.join(path, audio))

        try:
            torch.save({"lyrics": lyrics, "tokens": tokens}, cache_path)
            print("💾 saved cache:", cache_path)
        except Exception as e:
            print("⚠️ failed to save cache:", e)

    T, C = tokens.shape
    if T > segment_tokens:
        start = 0
        tokens = tokens[start : start + segment_tokens]
    elif T < segment_tokens:
        pad_len = segment_tokens - T
        pad = torch.full((pad_len, C), fill_value=0, dtype=tokens.dtype)
        tokens = torch.cat([tokens, pad], dim=0)

    lyrics = lyrics.to(device)
    tokens = tokens.unsqueeze(0).to(device)
    return lyrics.detach(), tokens.detach()

def load_batch(batch_size: int, segment_tokens: int):
    lyrics_list = []
    tokens_list = []
    for _ in range(batch_size):
        lyr, tok = load_random_song(segment_tokens)
        lyrics_list.append(lyr)
        tokens_list.append(tok)

    lyrics = torch.cat(lyrics_list, dim=0)
    tokens = torch.cat(tokens_list, dim=0)
    return lyrics, tokens


device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
CHECKPOINT_PATH = "checkpoints/train_dataset.pt"
DEVICE = device

text_encoder = TextEncoder(max_tokens=512).to(torch.device("cpu"))
audio_encoder = AudioEncoder(device="cpu")
VOCAB_PER_CB = audio_encoder.vocab_size
CODE_DIM = audio_encoder.model.quantizer.vq.layers[0]._codebook.weight.shape[1]
EMBED_DIM = 512
MAX_TOKENS = 8192
EPOCHS = 10000
LR = 1e-4
tokens2d = audio_encoder.encode("dataset/song_001/song_001.mp3")
N_CODEBOOKS = tokens2d.shape[1]

transformer = SoundTransformer(
    vocab_size=VOCAB_PER_CB,
    n_codebooks=N_CODEBOOKS,
    embed_dim=EMBED_DIM,
    codebook_dim=CODE_DIM,
    num_heads=4,
    num_layers=4,
    max_seq_len=MAX_TOKENS
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(transformer.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9,0.98))

if os.path.exists(CHECKPOINT_PATH):
    print(f"\U0001f501 Found checkpoint at {CHECKPOINT_PATH}, resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        transformer.load_state_dict(checkpoint["model_state_dict"])
    else:
        transformer.load_state_dict(checkpoint)

    if "optimizer_state_dict" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("⚠️ Checkpoint doesn't contain optimizer/scheduler state. Starting fresh for them.")
else:
    print("\U0001f195 No checkpoint found, starting from scratch.")

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
print("Starting training …")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    transformer.train()
    optimizer.zero_grad(set_to_none=True)
    segment_tokens = get_segment_tokens(epoch)
    lyr, x = load_batch(4, segment_tokens)

    preds = transformer(lyr, x)
    with torch.no_grad():
        cb_tables = [
            audio_encoder.model.quantizer.vq.layers[c]._codebook.weight.to(x.device)
            for c in range(N_CODEBOOKS)
        ]
        tgt = torch.stack([
            cb_tables[c][x[..., c]] for c in range(N_CODEBOOKS)
        ], dim=2)

    preds = preds[:, :-1]
    tgt = tgt[:, 1:]
    loss = criterion(preds, tgt)
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"[Epoch {epoch}] L2 loss: {loss.item():.6f}")
    del lyr, x, preds, tgt, loss
    torch.mps.empty_cache()
    gc.collect()

    if epoch % 20 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"📂  saved checkpoint → {CHECKPOINT_PATH}")

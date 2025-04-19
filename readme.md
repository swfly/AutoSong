# 🎵 Lyric-to-Music Transformer – Project Notes

## 🧠 Purpose

This project aims to explore whether a **GPT-style autoregressive transformer** can learn to generate **music audio conditioned only on full lyrics and genre**, similar in concept to **Suno**.

The system predicts audio **block-by-block** (e.g., using EnCodec tokens), using the **textual prompt as context** and previously generated audio blocks to model temporal coherence.

In order to fully leverage the creativity of transformer, the lyrics is always provided in complete as memory prompt: it lets transformer to decide about the timing and structure of the song.
```
            +----------------------+
            | User: Lyrics + Genre |
            +----------------------+
                       |
                       v
                +--------------+
                | TextEncoder  |
                +--------------+
                       |
                       v
             +-------------------+
             | Lyrics Embedding  |
             +-------------------+
                       |
                       v
   +-------------------------------------+
   |         SoundTransformer           |
   | (autoregressive decoder-only model) |
   +-------------------------------------+
        ^                    |
        |                    v
        |       +-----------------------+
        |       | Audio Tokenization    |
        |       |    (AudioEncoder)     |
        |       +-----------------------+
        |                    |
        |                    v
        +<---- Generated Tokens (1D) ----+
                       |
                       v
                +--------------+
                | AudioDecoder |  ← (same model)
                +--------------+
                       |
                       v
               +----------------+
               |   Output WAV   |
               +----------------+
```
---

## 🎯 Goal of Initial Prototype

To build a **research-grade minimal system** that can:

1. Encode lyrics and genre as a conditioning vector (via BERT or similar)
2. Encode audio into token sequences using Meta’s EnCodec
3. Train a decoder-only transformer to predict next EnCodec tokens given:
   - `[lyrics_embedding] + [previous_audio_tokens]`
4. Decode generated tokens into raw audio for listening

The aim is to validate whether lyrics and genre can meaningfully guide music generation, in terms of a full composition, via a GPT-style model.


## 🎼 Why Full-Context Matters
Unlike loop-based or short-sample music generators, this project is explicitly designed to model long-range musical structure — including:

- Verse–chorus alternation

- Global lyric-to-melody alignment

- Energy curves that follow emotional arcs

- Recurring motifs, phrasing, and tension-resolution patterns

To enable this, the system is trained on entire song spans (up to 18,000 EnCodec tokens, ≈15–20 seconds of music) in a single transformer context window. This design choice:

- Forces the model to learn compositional structure over time

- Preserves the temporal relationship between lyrics and musical phrasing

- Allows lyric-conditioned guidance across full sections of a song

While this demands significantly more VRAM and compute than local models, it is essential to the goal of coherent music generation from start to finish.

## Data Flow

- Text encoder generates lyric tokens.
- AudioEncoder generates a **complete** sequence of sound tokens using Meta's EnCodec.
- Transformer takes Text as "memory" and predicts n+1-th sound token using previous n sound tokens.


## 🧪 Formal Full-Context Training
This project includes a **formal training script** that performs full-sequence training with constant VRAM usage. The goal is to explore generalization from lyrics and genre across a dataset of diverse music tracks.

### 🔧 Script: `scripts/train_dataset.py`

This script:

- Samples random `(lyrics, audio)` pairs from a dataset directory
- Encodes lyrics into a fixed embedding (via `TextEncoder`)
- Tokenizes audio using Meta’s EnCodec (via `AudioEncoder`)
- Offsets and flattens multi-codebook EnCodec tokens into a 1D token sequence
- Trains a decoder-only transformer on full 18,000-token sequences using causal self-attention and cross-attention over the fixed lyrics embedding

It uses:

- `CrossEntropyLoss` over the full vocabulary (`n_codebooks × vocab_per_codebook`)
- Cosine learning rate schedule from `LR_MAX` to `LR_MIN`
- Periodic checkpointing to disk

---

### 📁 Dataset Structure

Expected layout of the dataset directory:

```plaintext
dataset/
├── song_001/
│   ├── lyrics.txt
│   └── audio.mp3
├── song_002/
│   ├── lyrics.txt
│   └── audio.flac
├── song_003/
│   ├── lyrics.txt
│   └── audio.mp3
⋮
```

- Each subdirectory is a sample
- Each must contain:
  - A `.txt` file with the lyrics (UTF-8 plain text)
  - A `.mp3` or `.flac` audio file

> ⚠️ Audio files are resampled to 24 kHz and tokenized using Meta’s EnCodec (24 kHz model).

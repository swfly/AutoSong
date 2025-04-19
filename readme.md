# 🎵 Lyric-to-Music Transformer – Project Notes

## 🧠 Purpose

This project aims to explore whether a **GPT-style autoregressive transformer** can learn to generate **music audio conditioned only on full lyrics and genre**, similar in concept to **Suno**.

The system predicts audio **block-by-block** (e.g., using EnCodec tokens), using the **textual prompt as context** and previously generated audio blocks to model temporal coherence.

In order to fully leverage the creativity of transformer, the lyrics is always provided in complete as memory prompt: it lets transformer to decide about the timing and structure of the song.
---

## 🎯 Goal of Initial Prototype

To build a **research-grade minimal system** that can:

1. Encode lyrics and genre as a conditioning vector (via BERT or similar)
2. Encode audio into token sequences using Meta’s EnCodec
3. Train a decoder-only transformer to predict next EnCodec tokens given:
   - `[lyrics_embedding] + [previous_audio_tokens]`
4. Decode generated tokens into raw audio for listening

The aim is to validate whether lyrics and genre can meaningfully guide music generation via a GPT-style model.

## ✅ Completed Components

- `/models/text_encoder.py`: Loads and encodes lyrics using pretrained BERT (HuggingFace)
- `/models/audio_encoder.py`: Wraps Meta EnCodec to tokenize and reconstruct audio
- `/models/sound_transformer.py`: GPT-style autoregressive decoder that predicts the next EnCodec token, conditioned on a fixed lyrics embedding and past audio tokens. Accepts a [lyrics_embedding] vector as external memory and autoregressively models temporal audio coherence via causal self-attention. Designed to support flat or interleaved token sequences from multiple EnCodec codebooks.
- `/scripts/test_overfit.py`: Trains SoundTransformer to overfit a short music sample (~30 seconds) using flattened EnCodec tokens from 2 codebooks and corresponding lyrics. Validates whether the model can learn alignment between lyrics and the structure of the original song.
- `/scripts/test_predict.py`: Sequentially predict and recover the overfitted song using the saved checkpoint to test the effect of learning.
- `/scripts/train_dataset.py`: Do the actual training, see below.
---

## 🧪 Formal Full-Context Training

In addition to overfitting a single example, this project includes a **formal training script** that performs full-sequence training with constant VRAM usage. The goal is to explore generalization from lyrics and genre across a dataset of diverse music tracks.

### 🔧 Script: `scripts/train_full_context.py`

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



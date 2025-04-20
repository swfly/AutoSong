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
2. Encode audio into token sequences using Meta’s EnCodec. Bandwith is set to 1.5, which leads to low quality, but fine since we want to check the overall flow of music
3. Train a decoder-only transformer to predict next EnCodec tokens given:
   - `[lyrics_embedding] + [previous_audio_tokens]`
4. Decode generated tokens into raw audio for listening

The aim is to validate whether lyrics and genre can meaningfully guide music generation, in terms of a full composition, via a GPT-style model.

## Current Situation

The overfit test passed: it can generate a complete song with the network.

~~However, the real training with a variety of songs (5 of them) with known lyrics (chinese) doesn't work: it always generates blank audio.
To see what's wrong here, I reduce the size to 8K (previously 18K) with bandwidth=3 to check if the quality of audio and the scope of learning matters.~~

~~It now generates looping "la la la" sound with 8K context size.~~

No, GPT-generated code is garbage. Need to re-do the network from scratch.

## 🧮 Transformer Mechanics – Mathematical Summary

This section explains the internal mechanics of the SoundTransformer using mathematical notation.

### 🔸 Input Embeddings

Each token in the flattened audio sequence is embedded as:

$$
\mathbf{e}_i = \text{token\_emb}(t_i) + \text{cb\_emb}(c_i) + \text{pos\_emb}(i)
$$

Where:
- \( t_i \) is the token index (e.g., 0–8191)
- \( c_i = \left\lfloor \frac{t_i}{V} \right\rfloor \) is the codebook index (with \( V \) = vocab size per codebook)
- \( i \) is the absolute position
- All embedding vectors lie in \( \mathbb{R}^d \)

---

### 🔸 Text Embedding ("Memory")

Lyrics are encoded using a frozen BERT model to produce token-level hidden states:

$$
\mathbf{L} = \text{BERT}(\text{lyrics}) \in \mathbb{R}^{T_{\text{lyr}} \times d'}
$$

Then projected to match the transformer’s embedding size:

$$
\mathbf{M} = \mathbf{L} \cdot \mathbf{W}_{\text{proj}} \in \mathbb{R}^{T_{\text{lyr}} \times d}
$$

This becomes the **memory** used in cross-attention.

---

### 🔸 Decoder Layer – Self + Cross Attention

Each decoder layer applies both causal self-attention and lyric-conditioned cross-attention:

#### 1. Causal Self-Attention

$$
Q = XW_Q^{\text{(self)}}, \quad K = XW_K^{\text{(self)}}, \quad V = XW_V^{\text{(self)}}
$$

$$
\text{SelfAttn}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d}} + \text{mask} \right)V
$$

#### 2. Cross-Attention over Lyrics (Memory)

$$
Q = XW_Q^{\text{(cross)}}, \quad K = MW_K^{\text{(cross)}}, \quad V = MW_V^{\text{(cross)}}
$$

$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d}} \right)V
$$

Here, \( M \) is the lyric embedding sequence, and each audio token dynamically attends to relevant lyric tokens.

---

### 🔸 Feedforward and Residual Layers

Each transformer layer wraps these operations in standard residual connections:

$$
X \leftarrow \text{LayerNorm}(X + \text{SelfAttn}(X)) \\
X \leftarrow \text{LayerNorm}(X + \text{CrossAttn}(X, M)) \\
X \leftarrow \text{LayerNorm}(X + \text{MLP}(X))
$$

Where MLP is a two-layer feedforward network with GELU activation.

---


## 🎼 Why Full-Context Matters
Unlike loop-based or short-sample music generators, this project is explicitly designed to model long-range musical structure — including:

- Verse–chorus alternation

- Global lyric-to-melody alignment

- Energy curves that follow emotional arcs

- Recurring motifs, phrasing, and tension-resolution patterns

To enable this, the system is trained on entire song spans (up to 18,000 EnCodec tokens, at bandwith=1.5: ≈120 seconds of music!) in a single transformer context window. This design choice:

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

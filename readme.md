# 🎵 Lyric-to-Music Transformer – Project Notes

## 🧠 Purpose

This project aims to explore whether a **GPT-style autoregressive transformer** can learn to generate **music audio conditioned only on full lyrics and genre**, similar in concept to **Suno**.

The system predicts audio **block-by-block** (e.g., using EnCodec tokens), using the **textual prompt as context** and previously generated audio blocks to model temporal coherence.

In order to fully leverage the creativity of transformer, the lyrics is always provided in complete as memory prompt: it lets transformer to decide about the timing and structure of the song.

This project also features AI-assisted development: I have no idea what a transformer is and I let AI did most of the work. While, I evetually learned all of them and corrected the code. Not a good experience.

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

Now can actually "learn" multi-channel generating.


## Data Flow

- Text encoder generates lyric tokens.
- AudioEncoder generates a **complete** sequence of sound tokens using Meta's EnCodec.
- Transformer takes Text as "memory" and predicts n+1-th sound token using previous n sound tokens: the naive autoregression style.
```
          ┌────────────────────┐
          │  User Input        │
          │  Lyrics + Genre    │
          └────────┬───────────┘
                   │
                   ▼
         ┌──────────────────────┐
         │ BERT Text Encoder    │
         │ (frozen, token-level │
         │  hidden states)      │
         └────────┬─────────────┘
                  │
       token-level embeddings ∈ ℝ^{T × d_bert}
                  │
                  ▼
      ┌─────────────────────────────┐
      │ Linear Projection to d_model│
      └────────┬────────────────────┘
               │
   memory prompt M ∈ ℝ^{T × d_model}
               │
               ▼
    ┌─────────────────────────────────────┐
    │     SoundTransformer (decoder)      │
    │                                     │
    │ ┌───────────────────────────────┐   │
    │ │  Step 1: Embed Audio Tokens   │   │
    │ │                               │   │
    │ │  token_id → token_emb ∈ ℝ^d   │   │
    │ │  codebook_id → cb_emb ∈ ℝ^d   │   │
    │ │  position_id → pos_emb ∈ ℝ^d  │   │
    │ └────────────┬──────────────────┘   │
    │              │                      │
    │              ▼                      │
    │      summed embedding x ∈ ℝ^{T × d} │
    │              │                      │
    │              ▼                      │
    │  ┌─────────────────────────────┐    │
    │  │  Transformer Decoder Layers │    │
    │  │                             │    │
    │  │ - Causal Self-Attention     │    │
    │  │ - Cross-Attn to M           │◄───┼─ memory from lyrics
    │  │ - MLP + LayerNorms          │    │
    │  └────────┬────────────────────┘    │
    │           │                         │
    │           ▼                         │
    │     projected logits ∈ ℝ^{T × C × V}
    └───────────┬─────────────────────────┘
                │
                ▼
    ┌────────────────────────────────────┐
    │  Predicted Audio Tokens (1D, C)    │
    └───────────┬────────────────────────┘
                │
                ▼
    ┌────────────────────────────────────┐
    │     EnCodec Audio Decoder          │
    │     (Meta EnCodec, pretrained)     │
    └───────────┬────────────────────────┘
                │
                ▼
         ┌─────────────────────┐
         │     Output WAV      │
         └─────────────────────┘

```

## 🧮 Transformer Mechanics – Mathematical Summary

This section explains the internal mechanics of the SoundTransformer using mathematical notation.

### 🔸 Input Embeddings

Each audio input is a sequence of discrete tokens from multiple codebooks (channels), represented as a tensor of shape \( [B, T, C] \), where:

- \( B \) is the batch size
- \( T \) is the number of time steps
- \( C \) is the number of codebooks (channels)

For each token at time \( t \) and channel \( c \), its embedding is computed as:

$$
\mathbf{e}_{t, c} = \text{token\_emb}(x_{t, c}) + \text{channel\_emb}(c) + \text{pos\_emb}(t)
$$

Where:

- \( x_{t, c} \in \{0, \dots, V - 1\} \) is the token ID at timestep \( t \), channel \( c \)
- \( \text{token\_emb} \in \mathbb{R}^{V \times d} \) is the shared embedding lookup table for token IDs
- \( \text{channel\_emb}(c) \in \mathbb{R}^d \) is a learned embedding for the codebook index \( c \)
- \( \text{pos\_emb}(t) \in \mathbb{R}^d \) is a learned positional embedding for time \( t \)
- All embeddings are in \( \mathbb{R}^d \), the transformer model dimension

The embeddings are computed and summed **per token per channel**, then reshaped to a flattened sequence \( [B, T \cdot C, d] \) before being processed by the decoder stack.

This multi-channel embedding formulation preserves:
- **Temporal ordering** across time steps (via positional embeddings)
- **Semantic separation** across codebooks (via channel embeddings)
- **Discrete token identity** (via token embeddings)


---

### 🔸 Text Embedding ("Memory")

Lyrics are encoded using a frozen BERT model to produce token-level hidden states:

$$
\mathbf{L} = \text{BERT}(\text{lyrics}) \in \mathbb{R}^{T_{\text{lyr}} \times d'}
$$


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

The lyrics embedding is first projected to match the transformer’s embedding size:

$$
\mathbf{M} = \mathbf{L} \cdot \mathbf{W}_{\text{proj}} \in \mathbb{R}^{T_{\text{lyr}} \times d}
$$

This becomes the **memory** (M) used in cross-attention.
$$
Q = XW_Q^{\text{(cross)}}, \quad K = MW_K^{\text{(cross)}}, \quad V = MW_V^{\text{(cross)}}
$$

$$
\text{CrossAttn}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d}} \right)V
$$

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

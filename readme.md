# 🎵 AutoSong – Lyric‑Driven Autoregressive Composition

## 🧠 Purpose

This project aims to explore whether a **GPT-style autoregressive transformer** can learn to generate **music audio conditioned only on full lyrics and genre**, similar in concept to **Suno**.

The system predicts audio **block-by-block** (e.g., using EnCodec tokens), using the **textual prompt as context** and previously generated audio blocks to model temporal coherence.

In order to fully leverage the creativity of transformer, the lyrics is always provided in complete as memory prompt: it lets transformer decide about the timing and structure of the song.

This project also features AI-assisted development: I have no idea what a transformer is, and I let AI do most of the work. While I eventually learned all of them and corrected the code. Not a good experience.

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

## 🧮 Transformer Mechanics — Updated Mathematical Summary

This section details the mathematical operations of the **SoundTransformer**, which autoregressively predicts audio token sequences conditioned on lyrics embeddings.

### 🔹 Input: Audio Tokens

Let the input audio be quantized into \( C \) discrete **codebooks**, each producing a sequence of \( T \) tokens. The input tensor has shape:

\[
\mathbf{X}_{\text{raw}} \in \mathbb{N}^{B \times T \times C}
\]

Where:
- \( B \): Batch size  
- \( T \): Time steps  
- \( C \): Number of codebooks (channels)  
- Each token \( x_{b,t,c} \in \{0, \dots, V - 1\} \), with \( V \) the vocabulary size per codebook

Each token is embedded as:

\[
\mathbf{e}_{t, c} = \text{Embed}(x_{t, c}) + \text{ChannelEmb}(c) + \text{PosEmb}(t)
\]

Resulting in:

\[
\mathbf{E} \in \mathbb{R}^{B \times T \times C \times d}
\]

This is reshaped into a flat sequence:

\[
\mathbf{E}' = \text{reshape}(\mathbf{E}) \in \mathbb{R}^{B \times (T \cdot C) \times d}
\]

---

### 🔹 Conditioning: Lyrics Embedding

Lyrics are encoded as a sequence of pinyin tokens using a frozen embedding table:

\[
\mathbf{L}_{\text{raw}} = \text{TextEncoder}(\text{lyrics}) \in \mathbb{N}^{B \times T_{\text{text}}}
\]

Which are embedded with position:

\[
\mathbf{L} = \text{PhonemeEmbed}(\mathbf{L}_{\text{raw}}) + \text{PosEmb}_{\text{text}}
\quad \in \mathbb{R}^{B \times T_{\text{text}} \times d}
\]

A learned linear projection aligns dimensions:

\[
\mathbf{M} = \mathbf{L} \cdot \mathbf{W}_{\text{proj}} \in \mathbb{R}^{B \times T_{\text{text}} \times d}
\]

This becomes the **cross-attention memory**.

---

### 🔹 Decoder Layer: Causal Self-Attention + Cross-Attention

Each decoder layer applies:

#### 1. Causal Self-Attention over Audio Tokens

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q^{\text{(self)}}, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K^{\text{(self)}}, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V^{\text{(self)}}
\]

\[
\text{SelfAttn}(\mathbf{X}) = \text{softmax} \left( \frac{\mathbf{QK}^\top}{\sqrt{d}} + \text{CausalMask} \right) \mathbf{V}
\]

#### 2. Cross-Attention to Lyrics (Memory)

\[
\mathbf{Q}_{\text{cross}} = \mathbf{X} \mathbf{W}_Q^{\text{(cross)}}, \quad
\mathbf{K}_{\text{cross}} = \mathbf{M} \mathbf{W}_K^{\text{(cross)}}, \quad
\mathbf{V}_{\text{cross}} = \mathbf{M} \mathbf{W}_V^{\text{(cross)}}
\]

\[
\text{CrossAttn}(\mathbf{X}, \mathbf{M}) = \text{softmax} \left( \frac{\mathbf{Q}_{\text{cross}} \mathbf{K}_{\text{cross}}^\top}{\sqrt{d}} \right) \mathbf{V}_{\text{cross}}
\]

---

### 🔹 Feedforward and Residual Processing

Each layer wraps attention and MLP in residual blocks:

\[
\mathbf{X} \leftarrow \text{LayerNorm}(\mathbf{X} + \text{SelfAttn}(\mathbf{X}))
\]

\[
\mathbf{X} \leftarrow \text{LayerNorm}(\mathbf{X} + \text{CrossAttn}(\mathbf{X}, \mathbf{M}))
\]

\[
\mathbf{X} \leftarrow \text{LayerNorm}(\mathbf{X} + \text{MLP}(\mathbf{X}))
\]

With:

\[
\text{MLP}(\mathbf{X}) = \text{Dropout}(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{X}))
\]

---

### 🔹 Output: Token Prediction with LoRA Adaptation

After \( L \) decoder layers:

\[
\mathbf{H} \in \mathbb{R}^{B \times T \times C \times d}
\]

Shared output projection:

\[
\text{logits}_{\text{shared}} = \mathbf{H} \cdot \mathbf{W}_{\text{shared}}^\top \in \mathbb{R}^{B \times T \times C \times V}
\]

Optional LoRA-style per-channel adaptation:

\[
\text{logits}_{\text{LoRA}}^{(c)} = (\mathbf{H}_{:, :, c, :} \cdot \mathbf{U}^{(c)}) \cdot \mathbf{V}^{(c)\top}
\]

Final logits:

\[
\text{logits}^{(c)} = \text{logits}_{\text{shared}}^{(c)} + \text{logits}_{\text{LoRA}}^{(c)}
\]

This provides channel-aware output with efficient low-rank tuning capacity.

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

## 🚧 New Direction: Two-Stage Generation with VAE & AR

To improve the structure and semantic coherence of generated music, we are shifting to a **two-stage architecture** that separates **latent content modeling** from **low-level audio generation**. This approach introduces a learned **VQ-VAE** to produce meaningful, compact latent representations, enabling a more structured and efficient autoregressive modeling stage.

---

### 🧱 Architecture Overview

1. **Stage 1: VQ-VAE**

    - The VQ-VAE compresses each audio segment into a pair of discrete latent codes:
        - One represents **general acoustic content** (e.g., instrumental texture)
        - The other can optionally represent **vocal content**, and may be zeroed during training to simulate instrumental-only inputs

    - During training, the VQ-VAE decoder reconstructs a segment using **contextual latent input**:
        ```
        [ z_prev_0, z_prev_1,  z_curr_0, z_curr_1,  z_next_0, z_next_1 ]
        ```

    - The VQ-VAE operates directly on **EnCodec token sequences** and learns to reconstruct them using a 6-block latent input context.

2. **Stage 2: Autoregressive Transformer**

    - A GPT-style transformer is trained to **autoregress over the VQ-VAE latent codes**, conditioned on full lyrics and genre.
    - This stage models high-level musical form, phrasing, and structural dependencies at the latent level.
    - Input: `[lyrics]` → Output: `[z₁, z₂, ..., zₖ]`
    - These predicted latents are then passed to the VQ-VAE decoder to generate full EnCodec sequences and ultimately audio.

---

### 🎯 Why This Design?

This hierarchical factorization addresses multiple limitations of the earlier flat token autoregression approach:

- **Latent Compression**: Reduces sequence length and entropy for the AR model
- **Semantic Abstraction**: Latent tokens are learned, not fixed EnCodec tokens, enabling meaningful reuse and structure
- **Decoupled Roles**:
    - VQ-VAE handles **acoustic realism** and local fidelity
    - Transformer handles **structure and lyrical alignment**

---

## 🔄 Discrete Transformer Autoencoder (Gumbel-Softmax Bottleneck)

This variant replaces the unstable VQ bottleneck with a **fully differentiable Transformer-based autoencoder**, using **discrete token prediction** as the compression mechanism.

Instead of learning a quantized latent codebook, the encoder outputs a **distribution over vocabulary tokens**, and the representation is obtained via **Gumbel-Softmax sampling** during training:

---

### 📥 Encoder (Transformer)

- **Input**: EnCodec tokens `x ∈ ℕ^{T × C}` (T time steps, C codebooks)
- **Embedding**: Token + Channel + Position
- **Core**: Transformer Encoder
- **Output**: `logits ∈ ℝ^{T × C × V}` (per-channel vocabulary logits)

During training, apply **Gumbel-Softmax** for differentiable sampling:

\[
\mathbf{z}_{t,c} = \text{GumbelSoftmax}(\text{logits}_{t,c}, \tau, \text{hard=True})
\]

At inference time, use `argmax(logits)` for hard token extraction.

---

### 📤 Decoder (Transformer)

- **Input**: Discrete token sequence `z ∈ ℕ^{T × C}` from encoder (argmax or Gumbel)
- **Embedding**: Same as encoder
- **Core**: Transformer Decoder
- **Output**: `\hat{x} ∈ ℝ^{T × C × V}`

---

### 🧪 Training Objective

- **Reconstruction Loss**:
  \[
  \mathcal{L}_{\text{recon}} = \text{CrossEntropy}(\hat{x}, x)
  \]

- **Optional Entropy Regularization**:
  Encourage diverse token usage across sequences:
  \[
  \mathcal{L}_{\text{entropy}} = -\sum p \log p \quad \text{(avg over batch)}
  \]

- **Final Loss**:
  \[
  \mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{entropy}} \cdot \mathcal{L}_{\text{entropy}}
  \]

---

### 🧠 Advantages of This Design

- ✅ Fully differentiable: avoids VQ backprop issues
- ✅ No codebook collapse or dead entries
- ✅ Easier to train and scale
- ✅ Integrates smoothly with autoregressive generation pipelines

---

### 🛠 Notes

- Gumbel-Softmax temperature `τ` can be annealed during training for sharper distributions.
- This is **not a VAE** and doesn't rely on latent priors — it's a **discrete bottleneck autoencoder** purely optimized via reconstruction.


---

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

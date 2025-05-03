# 🎵 AutoSong – Lyric-Driven Autoregressive Composition

## 🧠 Purpose  
AutoSong explores whether a **GPT-style autoregressive Transformer** can compose *complete* pieces of music directly from **full-song lyrics and a genre tag**.  
The model receives the entire lyric script up front, allowing it to decide global form (verse–chorus order, bridge placement, etc.) while it generates audio **continuously**.

## 🎯 Prototype Goals  
| Stage | What we demonstrate |
|-------|---------------------|
| **1. Text → Conditioning** | Encode lyrics & genre with a frozen BERT-like encoder → dense matrix \(M\). |
| **2. Audio → Continuous Latents** | Compress 256 × 256-bin **mel-spectrogram** blocks with a **convolutional autoencoder (AE)**.<br> • Encoder \(E\): \((T, F)\to (C,16,16)\)<br> • No quantization — latent tensors remain **real-valued**. |
| **3. Transformer Autoreg.** | A causal decoder predicts the next latent patch \(\hat z_{t+1}\) given past latents \(z_{\le t}\) **and** cross-attention to \(M\). |
| **4. Latent → Audio** | AE decoder reconstructs mel frames; Griffin–Lim inverts them to waveform. |

Success is measured by both objective reconstruction (L1 on mels) and **subjective musicality** of new lyrics-conditioned generations.

## 🔄 Data Flow


## 🧮 Transformer Mechanics

### Input sequence  
Latent patches are raster-ordered:

\[
z_{t}\in\mathbb R^{C\times16\times16}\quad\Longrightarrow\quad
\tilde z_t=\operatorname{reshape}(z_t)\in\mathbb R^{(C\!\cdot\!256)\times d}
\]

with positional \((\mathbf{p})\) and channel \((\mathbf{c})\) embeddings added.

### Layer stack  
For each block \(l=1..L\):

\[
\begin{aligned}
\mathbf X &\leftarrow \text{LayerNorm}\!\left(\mathbf X+\text{SelfAttn}_l(\mathbf X)\right) \\
\mathbf X &\leftarrow \text{LayerNorm}\!\left(\mathbf X+\text{CrossAttn}_l(\mathbf X,\mathbf M)\right) \\
\mathbf X &\leftarrow \text{LayerNorm}\!\left(\mathbf X+\text{MLP}_l(\mathbf X)\right)
\end{aligned}
\]

### Output regression  
Final linear head predicts next patch:

\[
\hat z_{t+1}=W_{\text{out}}\mathbf X_{t}^{\text{(last)}}\in\mathbb R^{C\times16\times16}
\]

Loss =\(\;\lambda_{\text{L1}}\|\hat z_{t+1}-z_{t+1}\|_1\;+\;\lambda_{\text{GAN}}\mathcal L_{\text{adv}}\).

## 🏗️ Latent Autoencoder

### Encoder \(E\)  
Convolutional residual stack ↓↓ to \((C,16,16)\).  
Each output channel is **unit-Gaussian-regularised** with a small prior loss  

\[
\mathcal L_{\text{prior}}=\alpha\; \mathbb E[z^2]
\]

to keep the latent distribution well-behaved for the Transformer.

### Bottleneck  
No quantisation — latents remain continuous, enabling **hi-fi reconstruction**.

### Decoder \(D\)  
Symmetric up-sampling residual blocks with InstanceNorm → mel frame~\((T,F)\).

Reconstruction loss  

\[
\mathcal L_{\text{mel}} = \|D(E(\text{mel}))-\text{mel}\|_1
\]

## 🚀 Full-Context Training

* **Segment length**: up to 256 mel frames (≈ 1.3 s).  
* **Curriculum**: start 64 frames → full 256.  
* **Batch** 4–16 (AMP enabled).  
* **Optimizer** AdamW, 1 e-4, cosine decay.  
* **Total loss**

\[
\mathcal L = \mathcal L_{\text{mel}} + \beta\,\mathcal L_{\text{prior}}
           + \gamma\,\mathcal L_{\text{adv}}
           + \delta\,\mathcal L_{\text{feature}}
\]

* **Adversary**: Patch-GAN on mel spectrograms to sharpen transients.

## 🗂️ Dataset Layout
```plaintext
dataset/
├── song_0001/
│   ├── lyrics.txt
│   └── audio.wav  (≥48 kHz mono)
├── song_0002/
│   ├── lyrics.txt
│   └── audio.wav
⋮
```
Lyrics are converted to pinyin-tone tokens; audio is resampled, converted to 256-bin mels, and cached.

## Research Log: The Road to Autoregressive Music Generation

1. I started with a pure end-to-end approach: using EnCodec + transformer autoregression.  
   It **didn't work at all**. Looking back, I think it's because EnCodec's latent codes are too high in information entropy, and doing end-to-end music generation is simply too ambitious — the structure is too deep and subtle.

2. So I tried scaling the problem down.  
   I thought, "Okay, what if we use an autoencoder to compress the content further, and let the transformer handle high-level composition instead?"  
   I attempted VQ-VAE on top of EnCodec, but that didn’t go well either. Discretizing EnCodec's output just wasn’t stable.

3. Then I realized: maybe we shouldn't compress so aggressively.  
   I switched to MEL spectrograms, aiming to model **continuous sound dynamics** instead. This was much more promising: the autoencoder worked.  
   But VQ-VAE still didn’t — not sure exactly why. I eventually gave up on quantization and stuck with a continuous latent space.  
   The audio was now represented as 2D latent patches. I used a GAN to sharpen the output and avoid blurry reconstructions from L1/L2 losses.  
   Visualizing the latents, I noticed they looked like downsampled MELs — but with 4 channels showing visibly different structures. That felt like real, meaningful representation.

4. Then came autoregression again.  
   Previously, I was focused on predicting token distributions — but that didn’t transfer to continuous data. So I redesigned the logic:  
   - Embeddings became *biases* added to the latent patches  
   - I downsampled them via a simple MLP  
   - Ran them through a transformer stack  
   - And then upsampled back to patches  
   Most importantly, I shifted to **residual prediction**, which offloads capacity and makes learning much more efficient.

Now, things finally look reasonable and actually work.  
We can consider scaling the network and training on more diverse tracks.

---

### Reflection

Machine learning and generative modeling are no joke.  
Everything feels obvious in hindsight, but incredibly hard when you're stuck.  
This process — of hitting walls, debugging ideas, and slowly forming intuition — is what makes research difficult and meaningful.

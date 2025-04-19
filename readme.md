# 🎵 Lyric-to-Music Transformer – Project Notes

## 🧠 Purpose

This project aims to explore whether a **GPT-style autoregressive transformer** can learn to generate **music audio conditioned only on full lyrics and genre**, similar in concept to **Suno**.

The system predicts audio **block-by-block** (e.g., using EnCodec tokens), using the **textual prompt as context** and previously generated audio blocks to model temporal coherence.

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
- `/scripts/train_overfit.py`: Trains SoundTransformer to overfit a short music sample (~30 seconds) using flattened EnCodec tokens from 2 codebooks and corresponding lyrics. Validates whether the model can learn alignment between lyrics and the structure of the original song.
---
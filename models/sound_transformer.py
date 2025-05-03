import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vocabulary as vocab_mod

# ────────────────────────── helpers ──────────────────────────

def build_causal_mask(t: int, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Upper‑triangular (t×t) mask with 0 on diag / -inf above it (causal). Cached on CPU by default."""
    return torch.triu(torch.full((t, t), float("-inf"), device=device, dtype=dtype), diagonal=1)

# ─────────────────────── building blocks ───────────────────────

class DecoderLayer(nn.Module):
    """Pre‑norm Transformer decoder block (SelfAttn ▶ CrossAttn ▶ MLP) with GELU."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # self‑attention (causal)
        y = self.ln1(x)
        x = x + self.self_attn(y, y, y, attn_mask=attn_mask, is_causal = True, need_weights=False)[0]
        # cross‑attention to text memory
        y = self.ln2(x)
        x = x + self.cross_attn(y, memory, memory, is_causal = False, need_weights=False)[0]
        # feed‑forward
        y = self.ln3(x)
        x = x + self.mlp(y)
        return x

# ────────────────────── Sound Transformer (autoregressive) ──────────────────────

class SoundTransformerContinuous(nn.Module):
    """Latent‑patch autoregressive Transformer with learned down‑ & up‑samplers.

    The public API (constructor + forward signature) is unchanged; internals are tweaked to
    reduce information bottlenecks while staying lightweight.
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_hw: Tuple[int, int] = (32, 32),
        d_model: int = 512,
        hidden_dim: int | None = None,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        max_text_len: int = 512,
    ):
        super().__init__()
        H, W = patch_hw
        self.C, self.H, self.W = in_channels, H, W


        # temporal positional embeddings (max_seq_len * N_grid tokens)
        self.chan_emb = nn.Embedding(in_channels, H*W)
        self.time_pos = nn.Parameter(torch.empty(1, max_seq_len, H*W))
        nn.init.normal_(self.time_pos, std=1.0)

        vocab = vocab_mod.generate_pinyin_vocab()
        self.text_emb  = nn.Embedding(len(vocab), d_model)
        self.text_pos = nn.Parameter(torch.empty(1, max_text_len, d_model))
        nn.init.normal_(self.text_pos, std=1.0)

        self.token_proj = nn.Sequential(
            nn.Linear(in_channels * H * W, 4 * d_model),  # scale up to 4 * d_model
            nn.GELU(),                                  # apply GELU non-linearity
            nn.Linear(4 * d_model, d_model)              # scale back down to d_model
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers))

        self.recon_proj = nn.Sequential(
            nn.Linear(d_model, 4 * in_channels * H * W),  # scale up to 4 * in_channels * H * W
            nn.GELU(),                                  # apply GELU non-linearity
            nn.Linear(4 * in_channels * H * W, in_channels * H * W)  # scale back down to original shape
        )

        self.delta_scale = nn.Parameter(torch.full((in_channels,), 0.1))

        # cache a big causal mask on CPU; slice at runtime (saves allocation)
        full_mask = build_causal_mask(max_seq_len)
        self.register_buffer("_causal_mask", full_mask, persistent=False)

    # ─────────────────────────── forward ────────────────────────────

    def forward(self, text_ids: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Args:
            text_ids: (B,T)   int64
            latents : (B,S,C,H,W) float32
        Returns:
            Reconstructed latents with identical shape.
        """
        B, S, C, H, W = latents.shape
        assert (H, W) == (self.H, self.W) and C == self.C, "latent shape mismatch"

        # latents are now [B, S, C, H, W]
        # apply channel embedding and time embedding to the latents
        latents_reshaped = latents.view(B, S, C, H * W)  # [B, S, C, H*W]
        chan_emb_exp = self.chan_emb.weight.view(1, 1, C, -1)
        latents_with_chan_emb = latents_reshaped + chan_emb_exp
        time_emb = self.time_pos[:, :S, :].view(1, S, 1, -1)
        latents_with_emb = latents_with_chan_emb + time_emb
        encoded_patches = latents_with_emb.view(B, S, C, H, W)
        tokens = encoded_patches.view(B,S, C*H*W)

        # Project the embedded vectors to target dimension
        tokens = self.token_proj(tokens)
        # text memory (add pos‑emb)
        mem = self.text_emb(text_ids) + self.text_pos[:, : text_ids.size(1)]
        # causal transformer
        L = tokens.size(1)
        mask = self._causal_mask[:L, :L].to(latents.device)
        for blk in self.blocks:
            tokens = blk(tokens, mem, mask)

        # Project tokens back to latent patch shape
        tokens = self.recon_proj(tokens)  # [B, S, C*H*W]

        # Reshape to [B, S, C, H, W]
        recon = tokens.view(B, S, C, H, W)
        return latents + recon * torch.relu(self.delta_scale.view(1,1,-1,1,1))

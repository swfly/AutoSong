# composer.py
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

class Composer(nn.Module):
    """Latent‑patch autoregressive Transformer with learned down‑ & up‑samplers.

    The public API (constructor + forward signature) is unchanged; internals are tweaked to
    reduce information bottlenecks while staying lightweight.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.5,
        max_seq_len: int = 180,
        max_text_len: int = 512,
        max_sentence_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model

        # temporal positional embeddings (max_seq_len * N_grid tokens)

        vocab = vocab_mod.generate_pinyin_vocab()
        vocab_size = len(vocab)
        self.text_emb  = nn.Embedding(vocab_size, d_model)
        self.text_pos = nn.Parameter(torch.empty(1, max_text_len, d_model))
        nn.init.normal_(self.text_pos, std=1.0)
        nn.init.normal_(self.text_emb.weight, std=1.0)

        self.block_pos = nn.Parameter(torch.empty(1, max_sentence_len, d_model))
        self.empty_tag_emb = nn.Parameter(torch.ones(1, d_model))
        self.length_emb = nn.Embedding(8, d_model)
        self.role_emb = nn.Embedding(16, d_model)
        self.time_emb = nn.Embedding(max_seq_len, d_model)

        nn.init.normal_(self.block_pos, std=1.0)
        nn.init.normal_(self.empty_tag_emb, std=1.0)
        nn.init.normal_(self.length_emb.weight, std=1.0)
        nn.init.normal_(self.role_emb.weight, std=1.0)
        nn.init.normal_(self.time_emb.weight, std=1.0) 

        self.block_proj = nn.Linear(5 * d_model, d_model)
        self.pre_atten_proj = nn.Linear(max_sentence_len * d_model, d_model)
        self.after_atten_proj = nn.Linear(d_model,max_sentence_len * d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers))

        # cache a big causal mask on CPU; slice at runtime (saves allocation)
        full_mask = build_causal_mask(max_seq_len)
        self.register_buffer("_causal_mask", full_mask, persistent=False)

        
        self.token_distribution_head = nn.Linear(d_model, vocab_size)
        self.empty_tag_head = nn.Linear(max_sentence_len * d_model, 1)
        self.length_head = nn.Linear(max_sentence_len * d_model, 8)
        self.role_head = nn.Linear(max_sentence_len * d_model, 16)

    def build_blocks(self, block_sentences, block_empty_tags, block_lengths, block_roles):

        B, S, T = block_sentences.shape  # B = batch size, S = number of blocks (steps)
        time_indices = torch.arange(S, device=block_sentences.device).unsqueeze(0).expand(B, -1)
        
        sentence_pos_emb = self.block_pos[:, :T, :]
        embedded_sentences = self.text_emb(block_sentences)  # Shape: [B, S, 32, d_model]
        embedded_sentences = embedded_sentences + sentence_pos_emb
        embedded_empty_tags = (block_empty_tags * self.empty_tag_emb).unsqueeze(-2)   # Shape: [B, S, d_model]
        embedded_lengths = self.length_emb(block_lengths)  # Shape: [B, S, d_model]
        embedded_roles = self.role_emb(block_roles)  # Shape: [B, S, d_model]
        
        embedded_empty_tags = embedded_empty_tags.expand(-1, -1, T, -1)  # Shape: [B, S, 32, d_model]
        embedded_lengths = embedded_lengths.expand(-1, -1, T, -1)  # Shape: [B, S, 32, d_model]
        embedded_roles = embedded_roles.expand(-1, -1, T, -1)  # Shape: [B, S, 32, d_model]

        time_emb = self.time_emb(time_indices)  # Shape: [B, S, d_model]
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, T, -1)  # Expand time embeddings to the same shape as other embeddings

        latent_blocks = torch.cat([embedded_sentences, embedded_empty_tags, embedded_lengths, embedded_roles, time_emb], dim=-1)

        latent_blocks = self.block_proj(latent_blocks)  # Shape: [B, S, 32, d_model]
        latent_blocks = latent_blocks.reshape(B, S, -1)
        return latent_blocks
    # ─────────────────────────── forward ────────────────────────────

    def forward(self, text_ids, block_sentences, block_empty_tags, block_lengths, block_roles):

        B, S, T = block_sentences.shape  # B = batch size, S = number of blocks (steps)
        # Project the embedded vectors to target dimension
        latent_blocks = self.build_blocks(block_sentences, block_empty_tags, block_lengths, block_roles) #[B,S * T, d_model]
        latent_blocks = self.pre_atten_proj(latent_blocks)
        # text memory (add pos‑emb)
        mem = self.text_emb(text_ids) + self.text_pos[:, : text_ids.size(1)]
        # causal transformer
        L = block_empty_tags.size(1)
        mask = self._causal_mask[:L, :L].to(text_ids.device)
        for blk in self.blocks:
            latent_blocks = blk(latent_blocks, mem, mask)

        latent_blocks = self.after_atten_proj(latent_blocks)

        token_distributions = self.token_distribution_head(latent_blocks.view((B,S,T,-1))).view((B, S, T, -1))
        empty_tags = self.empty_tag_head(latent_blocks.view(B,S,-1))
        lengths = self.length_head(latent_blocks.view(B,S,-1))
        roles = self.role_head(latent_blocks.view(B,S,-1))

        return token_distributions, empty_tags, lengths, roles

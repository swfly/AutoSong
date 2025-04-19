"""
SoundTransformer  –  decoder‑only GPT that now cross‑attends over the *entire*
lyrics sequence instead of a single vector.
"""

import torch
import torch.nn as nn


class CachingDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        # causal self‑attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        # cross‑attention over lyrics
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    # --------------------------- forward --------------------------- #
    def forward(
        self,
        x: torch.Tensor,             # [B, S_new, D]
        memory: torch.Tensor,        # [B, S_text, D]
        past_kv=None,
        *,
        use_cache: bool = False,
    ):
        # ── causal self‑attention ── #
        q = k = v = x
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], 1)
            v = torch.cat([pv, v], 1)

        attn_out, _ = self.self_attn(q, k, v, need_weights=False, is_causal=False)
        x = self.norm1(x + self.drop(attn_out))

        # ── cross‑attention over lyrics ── #
        ca_out, _ = self.cross_attn(
            x, memory, memory, need_weights=False
        )
        x = self.norm2(x + self.drop(ca_out))

        # ── MLP ── #
        x = self.norm3(x + self.drop(self.mlp(x)))

        if use_cache:
            return x, (k.detach(), v.detach())

        return x, None


class SoundTransformer(nn.Module):
    def __init__(
        self,
        vocab_size_per_cb: int,
        n_codebooks: int,
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        """
        Args
        ----
        vocab_size_per_cb : EnCodec vocab size for one codebook.
        n_codebooks       : number of EnCodec codebooks concatenated.
        """
        super().__init__()
        self.vocab_size_per_cb = vocab_size_per_cb
        self.n_codebooks       = n_codebooks

        # token‑type + position embeddings
        self.token_emb = nn.Embedding(vocab_size_per_cb * n_codebooks, embed_dim)
        self.cb_emb    = nn.Embedding(n_codebooks, embed_dim)
        self.pos_emb   = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # lyrics projection (matches embed_dim)
        self.lyrics_proj = nn.Linear(embed_dim, embed_dim)

        # stack of decoder layers
        self.layers = nn.ModuleList(
            CachingDecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        )

        self.out_proj    = nn.Linear(embed_dim, vocab_size_per_cb * n_codebooks)
        self.max_seq_len = max_seq_len

    # --------------------------- forward --------------------------- #
    def forward(
        self,
        lyrics_embed: torch.Tensor,  # [B, S_text, D]  (full sequence)
        token_ids: torch.Tensor,     # [B, S_new]
        *,
        past_kv=None,
        use_cache: bool = False,
        step: int = 0,
    ):
        """
        Args
        ----
        lyrics_embed : full‑sequence embeddings from TextEncoder.encode(...)
        token_ids    : newly fed audio tokens
        step         : starting position offset for pos_emb (needed when
                       generating incrementally with cache)
        """
        B, S_new = token_ids.shape

        # --- token/CB + position embeddings ------------------------ #
        cb_idx = token_ids // self.vocab_size_per_cb
        x = (
            self.token_emb(token_ids)
            + self.cb_emb(cb_idx)
            + self.pos_emb[:, step : step + S_new]
        )

        # --- projected lyric memory  ------------------------------- #
        # shape stays  [B, S_text, D]
        memory = self.lyrics_proj(lyrics_embed)

        # --- transformer layers ------------------------------------ #
        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = None if past_kv is None else past_kv[i]
            x, kv = layer(x, memory, pkv, use_cache=use_cache)
            if use_cache:
                new_cache.append(kv)

        logits = self.out_proj(x)  # [B, S_new, vocab]
        return (logits, new_cache) if use_cache else logits

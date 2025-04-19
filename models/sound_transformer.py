import torch
import torch.nn as nn


# ───────────────────────────── decoder layer ───────────────────────────── #
class CachingDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1, self.norm2, self.norm3 = (
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
        )
        self.drop = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,               # (B, 1, D) during gen
        memory: torch.Tensor,          # (B, 1, D)
        past_kv=None,                  # None | (k,v)
        *,
        use_cache: bool = False,
    ):
        q = k = v = x
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], 1)
            v = torch.cat([pv, v], 1)

        attn_out, _ = self.self_attn(q, k, v, need_weights=False, is_causal=False)
        x = self.norm1(x + self.drop(attn_out))

        ca_out, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + self.drop(ca_out))

        x = self.norm3(x + self.drop(self.mlp(x)))

        if use_cache:
            return x, (k.detach(), v.detach())
        return x, None


# ───────────────────────────── whole decoder ──────────────────────────── #
class SoundTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.lyrics_proj = nn.Linear(embed_dim, embed_dim)
        self.layers = nn.ModuleList(
            [CachingDecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------ #
    def forward(
        self,
        lyrics_embed: torch.Tensor,    # (B, D)
        token_ids: torch.Tensor,       # (B, S_new)
        *,
        past_kv=None,                  # None | list[(k,v)]
        use_cache: bool = False,
        step: int = 0,
    ):
        B, S_new = token_ids.shape
        x = self.token_emb(token_ids) + self.pos_emb[:, step : step + S_new]

        memory = self.lyrics_proj(lyrics_embed).unsqueeze(1)

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = None if past_kv is None else past_kv[i]
            x, kv = layer(x, memory, pkv, use_cache=use_cache)
            if use_cache:
                new_cache.append(kv)

        logits = self.out_proj(x)
        return (logits, new_cache) if use_cache else logits

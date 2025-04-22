import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import os, sys, random, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import models.vocabulary

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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


    def forward(self, x, memory, attn_mask=None, past_kv=None, *, use_cache=False):
        q = k = v = x
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], 1)
            v = torch.cat([pv, v], 1)

        attn_out, _ = self.self_attn(q, k, v, need_weights=False, is_causal=True, attn_mask = attn_mask)
        x = self.norm1(x + self.drop(attn_out))

        ca_out, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + self.drop(ca_out))

        x = self.norm3(x + self.drop(self.mlp(x)))
        return x, None

def build_time_causal_mask(max_t: int, n_codebooks: int,
                           device="cpu", dtype=torch.float32):
    idx = torch.arange(max_t * n_codebooks, device=device)
    t  = idx // n_codebooks
    mask = (t.unsqueeze(0) < t.unsqueeze(1))
    mask = mask.to(dtype).masked_fill(mask, float("-inf"))
    return mask

class SoundTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        embed_dim: int = 768,
        codebook_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim

        self.vocab = models.vocabulary.generate_pinyin_vocab()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.channel_emb = nn.Embedding(n_codebooks, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        self.phoneme_emb = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embed_dim)

        self.lyrics_proj = nn.Linear(embed_dim, embed_dim)

        self.layers = nn.ModuleList(
            DecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        )

        self.out_proj = nn.Linear(embed_dim, codebook_dim, bias=False)

        self.max_seq_len = max_seq_len
        self.register_buffer("channel_ids", torch.arange(n_codebooks))
        full_mask = build_time_causal_mask(max_seq_len, n_codebooks, dtype=torch.float32)
        self.register_buffer("time_causal_mask", full_mask)

    def forward(
        self,
        lyrics: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        past_kv=None,
        use_cache: bool = False,
        step: int = 0,
    ):
        B, S_new, C = token_ids.shape

        x = self.token_emb(token_ids.reshape(-1)).view(B, S_new, C, -1)

        lyrics_embed = self.phoneme_emb(lyrics)
        pos_lyrics = self.pos_emb[:, :lyrics_embed.size(1)]
        lyrics_embed = lyrics_embed + pos_lyrics

        channel_emb = self.channel_emb(self.channel_ids).unsqueeze(0).unsqueeze(0)
        pos_emb = self.pos_emb[:, step : step + S_new].unsqueeze(2)

        x = x + channel_emb + pos_emb
        memory = self.lyrics_proj(lyrics_embed)

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = None if past_kv is None else past_kv[i]
            flat_x = x.view(B, S_new * C, -1)
            seq_len = flat_x.size(1)
            attn_mask = self.time_causal_mask[:seq_len, :seq_len].to(flat_x.device, flat_x.dtype)
            x, kv = layer(flat_x, memory, attn_mask=attn_mask, past_kv=pkv, use_cache=use_cache)
            x = x.view(B, S_new, C, -1)
            if use_cache:
                new_cache.append(kv)

        vectors = self.out_proj(x)
        return (vectors, new_cache) if use_cache else vectors

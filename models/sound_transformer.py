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
        # Causal self-attention
        q = k = v = x
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], 1)
            v = torch.cat([pv, v], 1)

        attn_out, _ = self.self_attn(q, k, v, need_weights=False, is_causal=True, attn_mask = attn_mask)
        x = self.norm1(x + self.drop(attn_out))

        # Cross-attention over lyrics
        ca_out, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + self.drop(ca_out))

        # MLP
        x = self.norm3(x + self.drop(self.mlp(x)))
        return x, None

# --- helper -------------------------------------------------------------
def build_time_causal_mask(max_t: int, n_codebooks: int,
                           device="cpu", dtype=torch.float32):
    """
    Return a mask of shape [max_t * C, max_t * C] where entry (i,j) = 0
    iff the *time step* of j is ≤ that of i, otherwise -inf.
    """
    idx = torch.arange(max_t * n_codebooks, device=device)
    t  = idx // n_codebooks                      # time index of every position
    mask = (t.unsqueeze(0) < t.unsqueeze(1))     # True if j is strictly *future*
    mask = mask.to(dtype).masked_fill(mask, float("-inf"))
    return mask
class SoundTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,        # vocab size per channel (single codebook)
        n_codebooks: int,       # number of codebooks (channels)
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        """
        Args:
        ----
        vocab_size : The vocab size for a single codebook.
        n_codebooks: The number of codebooks (channels).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        
        self.vocab = models.vocabulary.generate_pinyin_vocab()
        # Embedding layers
        self.token_emb = nn.Embedding(vocab_size, embed_dim)  # Embedding for token IDs
        self.channel_emb = nn.Embedding(n_codebooks, embed_dim)  # Embedding for channels
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))  # Positional embeddings
        self.phoneme_emb = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embed_dim)

        # Lyrics projection (matches embed_dim)
        self.lyrics_proj = nn.Linear(embed_dim, embed_dim)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            DecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        )
        # Shared base projection
        self.shared_out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Per-channel low-rank adaptation
        self.lora_u = nn.Parameter(torch.zeros(n_codebooks, embed_dim))       # U ∈ [C, d]
        self.lora_v = nn.Parameter(torch.zeros(n_codebooks, vocab_size))      # V ∈ [C, V]
        nn.init.normal_(self.lora_u, std=1e-4)
        nn.init.normal_(self.lora_v, std=1e-4)
        
        # caches
        self.max_seq_len = max_seq_len
        self.register_buffer("channel_ids", torch.arange(n_codebooks))
        full_mask = build_time_causal_mask(max_seq_len,
                                                n_codebooks,
                                                dtype=torch.float32)
        self.register_buffer("time_causal_mask", full_mask)  # [T*C, T*C]
    def forward(
        self,
        lyrics: torch.Tensor,  # [B, S_text, D]  (full sequence)
        token_ids: torch.Tensor,     # [B, S, C], multi-channel input
        *,
        past_kv=None,
        use_cache: bool = False,
        step: int = 0,
    ):
        """
        Args
        ----
        lyrics : full‑sequence pinyin IDs from TextEncoder.encode(...)
        token_ids    : newly fed multi-channel audio tokens
        step         : starting position offset for pos_emb (needed when
                       generating incrementally with cache)
        """
        B, S_new, C = token_ids.shape  # Shape: [Batch, Sequence Length, Channels]

        # --- token/CB + position embeddings ------------------------ #
        # Token embedding per channel
        x = self.token_emb(token_ids.reshape(-1))  # Flatten the token_ids to [B * S * C]
        x = x.view(B, S_new, C, -1)  # Reshape back to [B, S_new, C, embed_dim]

        # Pinyin embedding + position embedding for lyrics
        lyrics_embed = self.phoneme_emb(lyrics)         # [B, S_text, 512]
        pos_lyrics = self.pos_emb[:, :lyrics_embed.size(1)]  # [1, S_lyrics, D]
        lyrics_embed = lyrics_embed + pos_lyrics  # broadcast add over batch
        # Channel embedding + position embedding
        channel_emb = self.channel_emb(self.channel_ids)
        channel_emb = channel_emb.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, C, embed_dim]

        pos_emb = self.pos_emb[:, step : step + S_new]  # [1, S_new, embed_dim]
        pos_emb = pos_emb.unsqueeze(2)  # Add channel dimension -> [1, S_new, C, embed_dim]

        x = x + channel_emb + pos_emb  # Adding token, channel, and position embeddings

        # --- projected lyric memory  ------------------------------- #
        
        memory = self.lyrics_proj(lyrics_embed)  # [B, S_text, embed_dim]

        # --- transformer layers ------------------------------------ #
        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = None if past_kv is None else past_kv[i]
            flat_x = x.view(B, S_new * C, -1)
            seq_len = flat_x.size(1)
            attn_mask = self.time_causal_mask[:seq_len, :seq_len].to(flat_x.device, flat_x.dtype)
            # def checkpoint_forward(x):
            #     return layer(x, memory, attn_mask=mask, past_kv=pkv, use_cache=use_cache)
            # x, kv = checkpoint(checkpoint_forward, flat_x, use_reentrant=False)
            x, kv = layer(flat_x, memory, attn_mask=attn_mask, past_kv=pkv, use_cache=use_cache)
            x = x.view(B, S_new, C, -1)  # Reshape back to [B, S_new, C, embed_dim]
            if use_cache:
                new_cache.append(kv)

        # --- Final output projection ------------------------------ #
        x_flat = x.view(B, S_new, C, -1)                       # [B, S, C, d]

        # Shared projection
        shared_logits = self.shared_out_proj(x_flat)           # [B, S, C, V]

        # LoRA projection: rank-1 residual per channel
        lora_logits = torch.einsum("bscd,cd->bsc", x_flat, self.lora_u)      # [B, S, C]
        lora_logits = lora_logits.unsqueeze(-1) * self.lora_v.unsqueeze(0).unsqueeze(0)  # [B, S, C, V]

        logits = shared_logits + lora_logits                   # [B, S, C, V]
        return (logits, new_cache) if use_cache else logits

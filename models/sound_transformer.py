import torch
import torch.nn as nn

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

        # Embedding layers
        self.token_emb = nn.Embedding(vocab_size, embed_dim)  # Embedding for token IDs
        self.channel_emb = nn.Embedding(n_codebooks, embed_dim)  # Embedding for channels
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))  # Positional embeddings

        # Lyrics projection (matches embed_dim)
        # 768 is the dimensionality of a bert token
        self.lyrics_proj = nn.Linear(768, embed_dim)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            DecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        )

        self.out_proj = nn.Linear(embed_dim, vocab_size)  # Per-channel output projection
        
        # caches
        self.max_seq_len = max_seq_len
        self.register_buffer("channel_ids", torch.arange(n_codebooks))
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        )
    def forward(
        self,
        lyrics_embed: torch.Tensor,  # [B, S_text, D]  (full sequence)
        token_ids: torch.Tensor,     # [B, S, C], multi-channel input
        *,
        past_kv=None,
        use_cache: bool = False,
        step: int = 0,
    ):
        """
        Args
        ----
        lyrics_embed : full‑sequence embeddings from TextEncoder.encode(...)
        token_ids    : newly fed multi-channel audio tokens
        step         : starting position offset for pos_emb (needed when
                       generating incrementally with cache)
        """
        B, S_new, C = token_ids.shape  # Shape: [Batch, Sequence Length, Channels]

        # --- token/CB + position embeddings ------------------------ #
        # Token embedding per channel
        x = self.token_emb(token_ids.reshape(-1))  # Flatten the token_ids to [B * S * C]
        x = x.view(B, S_new, C, -1)  # Reshape back to [B, S_new, C, embed_dim]

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
            mask = self.causal_mask[:seq_len, :seq_len].to(flat_x.device, flat_x.dtype)
            x, kv = layer(flat_x, memory, attn_mask=mask, past_kv=pkv, use_cache=use_cache)
            x = x.view(B, S_new, C, -1)  # Reshape back to [B, S_new, C, embed_dim]
            if use_cache:
                new_cache.append(kv)

        # --- Final output projection ------------------------------ #
        logits = self.out_proj(x.view(B, S_new * C, -1))  # [B, S_new * C, vocab_size]
        logits = logits.view(B, S_new, C, self.vocab_size)  # Reshape to [B, S_new, C, vocab_size]
        return (logits, new_cache) if use_cache else logits

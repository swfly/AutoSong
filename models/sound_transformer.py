import torch
import torch.nn as nn

class SoundTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,            # e.g., number of EnCodec tokens
        embed_dim: int = 768,       # size of transformer hidden state
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,    # including lyrics_embedding token(s)
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers)

        self.lyrics_proj = nn.Linear(embed_dim, embed_dim)  # optional: tune if lyrics dim != embed_dim
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, lyrics_embed, token_seq):
        """
        lyrics_embed: (batch, embed_dim)
        token_seq: (batch, seq_len) - EnCodec tokens
        """
        batch_size, seq_len = token_seq.shape

        # Embed tokens and positions
        token_emb = self.token_embedding(token_seq)               # (B, T, D)
        pos_emb = self.pos_embedding[:, :seq_len, :]              # (1, T, D)
        x = token_emb + pos_emb

        # Expand lyrics embedding to match sequence input
        lyrics_cond = self.lyrics_proj(lyrics_embed).unsqueeze(1) # (B, 1, D)

        # Causal mask for autoregression
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(token_seq.device)

        # Transformer decoder
        x = self.transformer(
            tgt=x,
            memory=lyrics_cond,    # acts as key/value for cross-attention
            tgt_mask=causal_mask
        )

        return self.output_proj(x)  # (B, T, vocab_size)

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
import math
# -------------------------------------------------------------------
# Discrete Transformer Autoencoder for EnCodec tokens
# -------------------------------------------------------------------

class DiscreteEncoder(nn.Module):
    """
    Transformer-based encoder that maps discrete audio tokens to logits over token vocab.
    Input: tokens [B, T, C]
    Output: logits [B, T, C, V]
    """
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.embed_dim = embed_dim

        # token, channel, position embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.channel_emb = nn.Embedding(n_codebooks, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Transformer encoder layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4*embed_dim,
                dropout=dropout,
                batch_first=True,
            ))
        self.encoder = nn.TransformerEncoder(nn.Sequential(*layers), num_layers=num_layers)

        # output projection to logits
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, C]
        B, T, C = tokens.shape
        # embed tokens
        x = self.token_emb(tokens.view(B, T*C))  # [B, T*C, D]
        x = x.view(B, T, C, self.embed_dim)
        # add channel+position
        ch = self.channel_emb(torch.arange(C, device=tokens.device))  # [C, D]
        ch = ch.unsqueeze(0).unsqueeze(0)  # [1,1,C,D]
        ps = self.pos_emb[:, :T, :].unsqueeze(2)  # [1,T,1,D]
        x = x + ch + ps
        # flatten channels into sequence
        x = x.view(B, T*C, self.embed_dim)
        # encode
        x = self.encoder(x)  # [B, T*C, D]
        # project to logits
        logits = self.out_proj(x)  # [B, T*C, V]
        logits = logits.view(B, T, C, self.vocab_size)
        return logits

class DiscreteDecoder(nn.Module):
    """
    Transformer-based decoder that reconstructs input tokens from discrete codes.
    Input: discrete tokens z [B, T, C]
    Output: logits [B, T, C, V]
    """
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        # similar embedding setup
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.channel_emb = nn.Embedding(n_codebooks, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Transformer decoder layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4*embed_dim,
                dropout=dropout,
                batch_first=True,
            ))
        self.decoder = nn.TransformerDecoder(nn.Sequential(*layers), num_layers=num_layers)

        # final projection
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward_embedded(self, z_emb: torch.Tensor) -> torch.Tensor:
        """
        z_emb: [B, T, C, D] — pre-embedded latent tokens
        returns logits [B, T, C, V]
        """
        B, T, C, D = z_emb.shape
        ch = self.channel_emb(torch.arange(C, device=z_emb.device)).unsqueeze(0).unsqueeze(0)  # [1,1,C,D]
        ps = self.pos_emb[:, :T, :].unsqueeze(2)  # [1,T,1,D]
        x = z_emb + ch + ps
        x = x.view(B, T*C, D)

        mask = torch.triu(torch.full((T*C, T*C), float('-inf'), device=x.device), diagonal=1)
        x = self.decoder(x, x, tgt_mask=mask)
        logits = self.out_proj(x).view(B, T, C, self.vocab_size)
        return logits
    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        # z_tokens: [B, T, C]
        B, T, C = z_tokens.shape
        # embed input tokens
        x = self.token_emb(z_tokens.view(B, T*C))
        x = x.view(B, T, C, -1)
        # add embeddings
        ch = self.channel_emb(torch.arange(C, device=z_tokens.device))  # [C, D]
        ch = ch.unsqueeze(0).unsqueeze(0)
        ps = self.pos_emb[:, :T, :].unsqueeze(2)
        x = x + ch + ps
        # flatten and prepare causal mask
        x = x.view(B, T*C, -1)
        # causal mask
        mask = torch.triu(torch.full((T*C, T*C), float('-inf'), device=x.device), diagonal=1)
        # decode
        x = self.decoder(x, x, tgt_mask=mask)
        logits = self.out_proj(x)
        logits = logits.view(B, T, C, self.vocab_size)
        return logits

class TransformerAutoencoder(nn.Module):
    """
    Discrete Transformer Autoencoder.
    encode -> argmax -> decode
    """
    def __init__(self, vocab_size, n_codebooks, **kwargs):
        super().__init__()
        self.encoder = DiscreteEncoder(vocab_size, n_codebooks, **kwargs)
        self.decoder = DiscreteDecoder(vocab_size, n_codebooks, **kwargs)

    def forward(self, tokens: torch.Tensor) -> dict:
        # tokens: [B, T, C]
        enc_logits = self.encoder(tokens)  # [B, T, C, V]
        # discrete bottleneck
        def gumbel_sample(logits, tau=1.0):
            return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        # One-hot vectors with gradients
        z_soft = gumbel_sample(enc_logits, tau=1.0)   # [B, T, C, V]
        z_tokens = z_soft.argmax(dim=-1)              # for logging only
        # project z_soft into embedding space manually
        emb_table = self.decoder.token_emb.weight  # [V, D]
        z_emb = torch.einsum("btcv,vd->btcd", z_soft, emb_table)  # [B, T, C, D]
        dec_logits = self.decoder.forward_embedded(z_emb)
        loss = F.cross_entropy(
            dec_logits.view(-1, dec_logits.size(-1)),
            tokens.view(-1),
        )
        return {
            "loss": loss,
            "enc_logits": enc_logits,
            "dec_logits": dec_logits,
            "z_tokens": z_tokens,
        }

# Example usage:
# ae = TransformerAutoencoder(vocab_size=1024, n_codebooks=8)
# out = ae(tokens)  # tokens: [B, T, C]


def chunk_encodec_tokens(tokens: torch.Tensor, seg_len: int) -> List[torch.Tensor]:
    """
    Split a (T, C) token grid into fixed-length segments, zero-padding the last.
    """
    T, C = tokens.shape
    num_seg = math.ceil(T / seg_len)
    pad = seg_len * num_seg - T
    if pad:
        pad_tensor = torch.zeros(pad, C, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, pad_tensor], 0)
    return list(tokens.view(num_seg, seg_len, C))

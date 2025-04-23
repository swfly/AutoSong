# vqvae_segment_minimal.py
# ──────────────────────────────────────────────────────────────────────────────
# Minimal autoencoder replacing VQ-VAE, keeping API of SegmentVQVAE intact.
# Segment shape: (256, 8)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Number of latent blocks per segment (unused in this minimal AE, kept for API)
POOL_SIZE = 8

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :].to(x.device)



class TransformerSegmentEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        emb_dim: int,
        latent_dim: int,
        n_latent_blocks: int
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim * n_codebooks)
        self.proj_in = nn.Linear(emb_dim * n_codebooks, latent_dim * 2)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim * 2, nhead=4, dim_feedforward=512),
            num_layers=1
        )

        self.pool = nn.AdaptiveAvgPool1d(n_latent_blocks)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = tokens.shape
        x = self.emb(tokens).view(B, T, -1)  # (B, T, C*emb)
        x = self.pos_enc(x)                 # (B, T, D)
        x = self.proj_in(x)                 # (B, T, 2D)
        x = x.permute(1, 0, 2)              # → (T, B, D)
        x = self.transformer(x)             # (T, B, 2D)
        x = x.permute(1, 2, 0)              # → (B, 2D, T)
        x = self.pool(x).permute(0, 2, 1)   # (B, N, 2D)
        return x.chunk(2, dim=-1)           # (B, N, D), (B, N, D)

class TransformerSegmentDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        latent_dim: int,
        n_latent_blocks: int,
        seg_len: int
    ):
        super().__init__()
        self.seg_len = seg_len
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks

        input_dim = 6 * latent_dim * n_latent_blocks
        hidden_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seg_len * n_codebooks * vocab_size),
        )

    def forward(
        self,
        z_prev_instru, z_prev_vocal,
        z_curr_instru, z_curr_vocal,
        z_next_instru, z_next_vocal
    ) -> torch.Tensor:
        B = z_curr_instru.size(0)
        z = torch.cat([
            z_prev_instru, z_prev_vocal,
            z_curr_instru, z_curr_vocal,
            z_next_instru, z_next_vocal
        ], dim=1)                   # (B, 6N, D)
        z = z.view(B, -1)           # (B, 6ND)
        out = self.fc(z)            # (B, T*C*V)
        out = out.view(B, self.seg_len, self.n_codebooks, self.vocab_size)
        return out                  # (B, T, C, V)


class SegmentVQVAE(nn.Module):
    """
    Minimal autoencoder for EnCodec-token segments.
    Same class & method names as before, so you can swap it in directly.
    """
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        seg_len: int,
        block_pairs:int = 2,
        num_codes: int = 1024,
        emb_dim: int = 128,
        latent_dim: int = 256,
        beta: float = 0.1,
    ):
        super().__init__()
        self.seg_len     = seg_len
        self.n_codebooks = n_codebooks
        self.vocab_size  = vocab_size
        self.pair = block_pairs

        self.encoder = TransformerSegmentEncoder(vocab_size=vocab_size, n_codebooks=n_codebooks,
                            emb_dim=emb_dim, latent_dim=latent_dim, n_latent_blocks=block_pairs)
        self.decoder = TransformerSegmentDecoder(vocab_size=vocab_size, n_codebooks=n_codebooks,
                            latent_dim=latent_dim, n_latent_blocks=block_pairs, seg_len=seg_len)


        # classifier: 1×1 conv to vocab logits
        self.classifier = nn.Conv1d(emb_dim * n_codebooks, vocab_size * n_codebooks, kernel_size=1)


    def forward(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        *,
        zero_second_prob: float = 0.0,
        variance_loss_scale = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        tokens_*: (B, seg_len, C)
        Returns:
          total_loss, { recon_loss, vq_loss, diversity_loss, total }
        """
        B, T, C = tokens_curr.shape

        # --- encode all three segments ---
        z_prev_instru, z_prev_vocal = self.encoder(tokens_prev)   # (B, N, D)
        z_curr_instru, z_curr_vocal = self.encoder(tokens_curr)
        z_next_instru, z_next_vocal = self.encoder(tokens_next)

        # --- maybe zero z_curr_vocal ---
        # if self.training and zero_second_prob > 0.0:
        #     if torch.rand(1).item() < zero_second_prob:
        #         z_curr_vocal = torch.zeros_like(z_curr_vocal)

        # --- concatenate latent blocks from all three segments ---
        logits = self.decoder(
            z_prev_instru, z_prev_vocal,
            z_curr_instru, z_curr_vocal,
            z_next_instru, z_next_vocal
        )


        # --- reconstruction loss ---
        recon_loss = F.cross_entropy(
            logits.reshape(B * T * C, self.vocab_size),
            tokens_curr.reshape(B * T * C).long(),
            reduction="mean"
        )

        # --- metrics (no VQ, no diversity) ---
        vq_loss        = torch.tensor(0.0, device=recon_loss.device)
        diversity_loss = torch.tensor(0.0, device=recon_loss.device)
        total_loss     = recon_loss

        metrics = {
            "recon_loss":   recon_loss.detach(),
            "vq_loss":      vq_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "total":        total_loss.detach(),
        }
        return total_loss, metrics

# ──────────────────────────── utility: chopper ────────────────────────────────

def chunk_encodec_tokens(tokens: torch.Tensor, seg_len: int) -> List[torch.Tensor]:
    """Unchanged helper to split a long EnCodec-token grid into fixed-length segments."""
    T, C = tokens.shape
    num_seg = math.ceil(T / seg_len)
    pad = seg_len * num_seg - T
    if pad:
        pad_tensor = torch.zeros(pad, C, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, pad_tensor], 0)
    return list(tokens.view(num_seg, seg_len, C))
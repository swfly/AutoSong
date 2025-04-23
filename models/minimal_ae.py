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
        num_codes: int = 1024,
        emb_dim: int = 128,
        latent_dim: int = 256,
        beta: float = 0.1,
    ):
        super().__init__()
        self.seg_len     = seg_len
        self.n_codebooks = n_codebooks
        self.vocab_size  = vocab_size

        # token embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # encoder: conv → latent features (B, latent_dim, T)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(emb_dim * n_codebooks, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # decoder: convtranspose → reconstruct embedding space
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(latent_dim, emb_dim * n_codebooks, kernel_size=3, padding=1),
        )

        # classifier: 1×1 conv to vocab logits
        self.classifier = nn.Conv1d(emb_dim * n_codebooks, vocab_size * n_codebooks, kernel_size=1)

    def encode_blocks(
        self,
        tokens: torch.Tensor,
        zero_second: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stub to match original API.
        Returns:
          z_e    : (B, latent_dim, T)
          codes  : dummy tensor of shape (B, POOL_SIZE)
          vq_loss: zero
        """
        B, T, C = tokens.shape
        x = self.embedding(tokens.long())                    # (B, T, C, emb)
        x = x.view(B, T, -1).permute(0, 2, 1)                # (B, emb*C, T)
        z = self.encoder_conv(x)                             # (B, latent_dim, T)
        codes   = torch.zeros(B, POOL_SIZE, dtype=torch.long, device=tokens.device)
        vq_loss = torch.zeros(1, device=tokens.device)
        return z, codes, vq_loss

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
        # --- encode current segment ---
        B, T, C = tokens_curr.shape
        x = self.embedding(tokens_curr.long())                # (B, T, C, emb)
        x = x.view(B, T, -1).permute(0, 2, 1)                # (B, emb*C, T)
        z = self.encoder_conv(x)                             # (B, latent_dim, T)

        # --- decode back to logits ---
        h = self.decoder_conv(z)                             # (B, emb*C, T)
        logits = self.classifier(h)                          # (B, vocab_size*C, T)
        # reshape → (B, T, C, V)
        logits = logits.view(B, self.n_codebooks, self.vocab_size, T) \
                       .permute(0, 3, 1, 2)                 # (B, T, C, V)

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
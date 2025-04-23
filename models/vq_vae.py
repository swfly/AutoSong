# ──────────────────────────────────────────────────────────────────────────────
#  vqvae_segment.py
# ------------------------------------------------------------------------------
#  Custom VQ-VAE for EnCodec-token sequences, with EMA codebook & 8 latents.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────── constants ─────────────────────────────

# Number of latent blocks per segment (was 2 → now 8)
POOL_SIZE = 8


# ─────────────────────── helper: EMA VQ (VQ-VAE v2) ─────────────────────────

class VectorQuantizerEMA(nn.Module):
    """
    EMA codebook quantiser (à-la VQ-VAE v2).

    Args
    ----
    num_codes : vocabulary size (K)
    dim       : embedding dimension (D)
    beta      : commitment weight (β)
    decay     : EMA decay for codebook updates
    eps       : small constant to avoid divide-by-zero
    """
    def __init__(
        self,
        num_codes: int,
        dim: int,
        beta: float = 0.1,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.K, self.D = num_codes, dim
        self.beta, self.decay, self.eps = beta, decay, eps

        # Codebook as a parameter tensor, not Embedding
        self.codebook = nn.Parameter(torch.randn(num_codes, dim))
        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", torch.randn_like(self.codebook))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_e : (B, L, D) continuous encoder output
        Returns:
          z_q_st  : (B, L, D) quantised with straight-through
          codes   : (B, L)   code indices
          vq_loss : scalar   commitment loss only (codebook updated via EMA)
        """
        B, L, D = z_e.shape
        # compute squared-L2 distances to each code
        # dist: (B, L, K)
        dist = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2 * z_e @ self.codebook.t()
            + self.codebook.pow(2).sum(-1)
        )
        codes = dist.argmin(-1)  # (B, L)
        z_q = F.embedding(codes, self.codebook)  # (B, L, D)

        # EMA codebook update
        if self.training:
            # one-hot assignments: (B, L, K)
            one_hot = F.one_hot(codes, self.K).type_as(z_e)   # (B, L, K)
            flat_one_hot = one_hot.reshape(-1, self.K)           # (B*L, K)
            flat_z_e = z_e.reshape(-1, self.D)                   # (B*L, D)

            n_i = flat_one_hot.sum(dim=0)                     # (K,)
            dw  = flat_one_hot.T @ flat_z_e                   # (K, D)

            self.ema_cluster_size.mul_(self.decay).add_(n_i, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.ema_cluster_size + self.eps
            self.codebook.data.copy_(self.ema_w / n.unsqueeze(1))

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        # only commit loss: encourages z_e → code
        commit_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * commit_loss
        return z_q_st, codes, vq_loss


# ───────────────────────────── encoder / decoder ──────────────────────────────

class SegmentEncoder(nn.Module):
    """
    1-D CNN encoder mapping (B, T, C) token IDs → (B, POOL_SIZE, D) latents.
    """
    def __init__(
        self,
        vocab_per_codebook: int,
        n_codebooks: int,
        token_emb_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_per_codebook, token_emb_dim)
        in_dim = token_emb_dim * n_codebooks

        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, C)
        returns: z_e (B, POOL_SIZE, D)
        """
        B, T, C = tokens.shape
        x = self.token_emb(tokens.long())                  # (B, T, C, De)
        x = x.view(B, T, -1).transpose(1, 2)               # → (B, De*C, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # functional pooling handles MPS more robustly
        x = F.adaptive_avg_pool1d(x, POOL_SIZE)            # → (B, hidden_dim, POOL_SIZE)
        x = self.conv3(x)                                  # → (B, latent_dim, POOL_SIZE)
        # transpose to (B, POOL_SIZE, latent_dim)
        return x.transpose(1, 2)


class SegmentDecoder(nn.Module):
    """
    Decoder that takes 3 segments × POOL_SIZE latents each → reconstructs current tokens.
    """
    def __init__(
        self,
        n_codebooks: int,
        vocab_size: int,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        recon_length: int = 100,
    ):
        super().__init__()
        self.recon_length = recon_length
        self.n_codebooks  = n_codebooks

        # input to fc is (POOL_SIZE * 3) latent vectors of dim latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * POOL_SIZE * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, vocab_size * n_codebooks, kernel_size=1),
        )

    def forward(
        self,
        z_prev: torch.Tensor,   # (B, POOL_SIZE, D)
        z_curr: torch.Tensor,   # (B, POOL_SIZE, D)
        z_next: torch.Tensor    # (B, POOL_SIZE, D)
    ) -> torch.Tensor:
        """
        Returns logits over vocab: (B, T_seg, C, V)
        """
        B = z_curr.size(0)
        # concatenate along the "block" dimension → (B, 3*POOL_SIZE, D)
        z = torch.cat([z_prev, z_curr, z_next], dim=1)
        # flatten all latents → (B, 3*POOL_SIZE*D)
        h = self.fc(z.flatten(start_dim=1))               # (B, H)
        # expand in time and deconv → (B, V*C, T_seg)
        h = h.unsqueeze(-1).expand(-1, -1, self.recon_length)
        out = self.deconv(h)                              # (B, V*C, T_seg)
        # reshape → (B, T_seg, C, V)
        B, VC, T = out.shape
        out = out.view(B, self.n_codebooks, -1, T).permute(0, 3, 1, 2)
        return out


# ─────────────────────────── VQ-VAE wrapper module ────────────────────────────

class SegmentVQVAE(nn.Module):
    """
    VQ-VAE that chunks a long EnCodec token sequence into segments of length seg_len.
    Uses EMA quantizer and POOL_SIZE latents per segment.
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
        self.seg_len      = seg_len
        self.n_codebooks  = n_codebooks

        # encoder → (B, POOL_SIZE, latent_dim)
        self.encoder = SegmentEncoder(
            vocab_per_codebook=vocab_size,
            n_codebooks=n_codebooks,
            token_emb_dim=emb_dim,
            latent_dim=latent_dim,
        )
        # EMA quantizer
        self.vq = VectorQuantizerEMA(
            num_codes=num_codes,
            dim=latent_dim,
            beta=beta,
            decay=0.99,
            eps=1e-5,
        )
        # decoder reconstructs only the “current” segment
        self.decoder = SegmentDecoder(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            recon_length=seg_len,
        )

    def encode_blocks(
        self,
        tokens: torch.Tensor,
        zero_second: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tokens: (B, T_seg, C)
        Returns z_q: (B, POOL_SIZE, D), codes: (B, POOL_SIZE), vq_loss
        """
        z_e = self.encoder(tokens)                # (B,POOL_SIZE,D)
        z_q, codes, vq_loss = self.vq(z_e)        # EMA updates happen here
        if zero_second:
            # zero out the second block _index_ if you want vocal-drop
            z_q[:, 1].zero_()
        return z_q, codes, vq_loss

    def forward(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        *,
        zero_second_prob: float = 0.2,
        variance_loss_scale = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        tokens_*: (B, seg_len, C)
        Returns
        -------
        total_loss , { recon_loss, vq_loss, total }
        """
        # decide instrumental regime
        #zero_flag = torch.rand(()) < zero_second_prob
        zero_flag = False

        z_prev, _, vq_prev = self.encode_blocks(tokens_prev, zero_flag)
        z_curr, _, vq_curr = self.encode_blocks(tokens_curr, zero_flag)
        z_next, _, vq_next = self.encode_blocks(tokens_next, zero_flag)

        logits = self.decoder(z_prev, z_curr, z_next)  # (B, seg_len, C, V)

        B, T, C, V = logits.shape
        recon_loss = F.cross_entropy(
            logits.reshape(B*T*C, V),
            tokens_curr.reshape(B*T*C).long(),
            reduction='mean'
        )
        vq_loss = vq_prev + vq_curr + vq_next

        centered = self.vq.codebook - self.vq.codebook.mean(dim=0, keepdim=True)
        diversity_loss = -centered.pow(2).mean()
        total   = recon_loss + vq_loss + variance_loss_scale * diversity_loss

        metrics = {
            "recon_loss": recon_loss.detach(),
            "vq_loss":    vq_loss.detach(),
            "diversity_loss": -diversity_loss.detach(),
            "total":      total.detach(),
        }
        return total, metrics


# ──────────────────────────── utility: chopper ────────────────────────────────

def chunk_encodec_tokens(tokens: torch.Tensor, seg_len: int) -> List[torch.Tensor]:
    """
    Split long (T, C) token grid into segments of length seg_len.
    Pads the last chunk with zeros if needed.
    """
    T, C = tokens.shape
    num_seg = math.ceil(T / seg_len)
    pad     = seg_len * num_seg - T
    if pad:
        pad_tensor = torch.zeros(pad, C, dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, pad_tensor], dim=0)
    return list(tokens.view(num_seg, seg_len, C))
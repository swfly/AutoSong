# vqvae_segment_hybrid.py
# ──────────────────────────────────────────────────────────────────────────────
# Hybrid VQ-VAE:   • CNN encoder  → VQ bottleneck
#                  • Transformer decoder (context-aware)
# Keeps the same public API (SegmentVQVAE.forward, etc.).
# Segment tokens: shape (256, 8)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── quantiser ──────────────────────────
class VectorQuantizer(nn.Module):
    """
    Straight-through VQ layer (no EMA).
    """
    def __init__(self, num_codes: int, dim: int, beta: float = 0.1):
        super().__init__()
        self.K, self.D, self.beta = num_codes, dim, beta
        self.embed = nn.Embedding(num_codes, dim)
        self.embed.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z : (B, N, D)  latent grid
        returns:
          z_q_st : quantised with straight-through          (B, N, D)
          codes  : code indices                             (B, N)
          loss   : commitment loss (β · ‖z−sg(z_q)‖²)       ()
        """
        B, N, D = z.shape
        flat = z.reshape(-1, D)                                              # (B·N, D)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )                                                                    # (B·N, K)
        codes = dist.argmin(1)                                               # (B·N)
        z_q = self.embed(codes).view(B, N, D)                                # (B, N, D)

        commit = F.mse_loss(z, z_q.detach())
        loss = self.beta * commit
        z_q_st = z + (z_q - z).detach()
        return z_q_st, codes.view(B, N), loss


# ───────────────────────── CNN encoder ────────────────────────
class ConvSegmentEncoder(nn.Module):
    """
    Lightweight local encoder: tokens → (z_instru , z_vocal)
                               shape : (B, N, D) each
    """
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        emb_dim: int,
        latent_dim: int,
        n_latent_blocks: int,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        in_dim = emb_dim * n_codebooks
        hid = latent_dim * 4

        self.conv_feat = nn.Sequential(
            nn.Conv1d(in_dim, hid, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hid, 2 * latent_dim, 1),  # → 2D channels
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(n_latent_blocks)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens : (B, 256, C)
        """
        B, T, C = tokens.shape
        x = self.emb(tokens).view(B, T, -1).transpose(1, 2)   # (B, C·emb , T)
        x = self.conv_feat(x)                                 # (B, 2D , T)
        x = self.pool(x).transpose(1, 2)                      # (B, N , 2D)
        return x.chunk(2, dim=-1)                             # (B,N,D) ×2


# ───────────────────── transformer decoder ────────────────────
class TransformerSegmentDecoder(nn.Module):
    """
    Takes SIX quantised streams (prev/cur/next × instru/vocal) and
    reconstructs current-segment logits.
    """
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        latent_dim: int,
        n_latent_blocks: int,
        seg_len: int,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks

        d_model = latent_dim
        self.pos = nn.Parameter(torch.randn(1, 6 * n_latent_blocks, d_model))

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.proj = nn.Linear(6 * n_latent_blocks * d_model,
                              seg_len * n_codebooks * vocab_size)

    def forward(
        self,
        z_prev_i, z_prev_v,
        z_curr_i, z_curr_v,
        z_next_i, z_next_v
    ):
        """
        Each z_xx_* : (B, N, D)
        returns logits (B, T, C, V)
        """
        B = z_curr_i.size(0)
        z = torch.cat(
            [z_prev_i, z_prev_v, z_curr_i, z_curr_v, z_next_i, z_next_v],
            dim=1
        )                                            # (B, 6N, D)
        z = z + self.pos[:, : z.size(1), :]          # add learned pos
        z = self.tf(z.permute(1, 0, 2)).permute(1, 0, 2)  # (B, 6N, D)
        out = self.proj(z.flatten(1))                       # (B, T·C·V)
        out = out.view(B, self.seg_len, self.n_codebooks, self.vocab_size)
        return out


# ───────────────────────── VQ-VAE wrapper ─────────────────────
class SegmentVQVAE(nn.Module):
    """
    CNN encoder → two VQ codebooks → Transformer decoder.
    API unchanged from previous versions.
    """
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        seg_len: int,
        block_pairs: int = 2,
        num_codes: int = 512,
        emb_dim: int = 128,
        latent_dim: int = 128,
        beta: float = 0.05,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks

        self.encoder = ConvSegmentEncoder(
            vocab_size, n_codebooks, emb_dim, latent_dim, block_pairs
        )
        self.vq_instru = VectorQuantizer(num_codes, latent_dim, beta)
        self.vq_vocal  = VectorQuantizer(num_codes, latent_dim, beta)

        self.decoder = TransformerSegmentDecoder(
            vocab_size, n_codebooks, latent_dim, block_pairs, seg_len
        )

    # ─────────────────── forward ────────────────────
    def forward(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        *,
        zero_second_prob: float = 0.0,
        variance_loss_scale: float = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        B, T, C = tokens_curr.shape

        # encode three segments
        p_i, p_v = self.encoder(tokens_prev)
        c_i, c_v = self.encoder(tokens_curr)
        n_i, n_v = self.encoder(tokens_next)

        # quantise
        p_i_q, _, vq_pi = self.vq_instru(p_i)
        c_i_q, _, vq_ci = self.vq_instru(c_i)
        n_i_q, _, vq_ni = self.vq_instru(n_i)

        p_v_q, _, vq_pv = self.vq_vocal(p_v)
        c_v_q, _, vq_cv = self.vq_vocal(c_v)
        n_v_q, _, vq_nv = self.vq_vocal(n_v)

        # optional vocal drop on current segment
        if self.training and zero_second_prob and torch.rand(()) < zero_second_prob:
            c_v_q = torch.zeros_like(c_v_q)

        logits = self.decoder(
            p_i_q, p_v_q,
            c_i_q, c_v_q,
            n_i_q, n_v_q
        )  # (B, T, C, V)

        recon_loss = F.cross_entropy(
            logits.reshape(B * T * C, self.vocab_size),
            tokens_curr.reshape(B * T * C).long(),
            reduction="mean"
        )

        vq_loss = vq_pi + vq_ci + vq_ni + vq_pv + vq_cv + vq_nv
        total_loss = recon_loss + vq_loss

        return total_loss, {
            "recon_loss": recon_loss.detach(),
            "vq_loss":    vq_loss.detach(),
            "total":      total_loss.detach(),
        }


# ───────────────────────── utility ────────────────────────────
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

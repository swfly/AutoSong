from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantizer takes [B, L_l, D_l] and 
# tries to find a single-channel token to best represent it;
# a quick way to have multi-channel tokens is to have multiple quantizers.
class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int, beta: float = 0.1,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K = vocab_size
        self.D = latent_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        self.embed = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embed.weight, -1.0 / self.K, 1.0 / self.K)

        # EMA buffers
        self.register_buffer('ema_cluster_size', torch.zeros(self.K))
        self.register_buffer('ema_w', self.embed.weight.data.clone())

    def forward(self, z: torch.Tensor):
        # z: (B, N, D)
        B, N, D = z.shape
        flat_z = z.reshape(-1, D)  # (B*N, D)

        # Compute distances to codebook
        # dist: (B*N, K)
        dist = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(dim=1)
        )
        # Get codes
        codes = dist.argmin(dim=1)  # (B*N,)
        z_q = self.embed(codes).view(B, N, D)

        # EMA codebook update
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(codes, self.K).type_as(flat_z)  # (B*N, K)
                # Count and sums
                cluster_size = one_hot.sum(dim=0)  # (K,)
                dw = one_hot.t() @ flat_z          # (K, D)

                # Update EMA variables
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_w.mul_(self.decay).add_(dw,           alpha=1 - self.decay)

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size + self.eps
                # Normalize ema_w and update embedding weight
                updated_weight = self.ema_w / n.unsqueeze(1)
                self.embed.weight.data.copy_(updated_weight)

        # Commitment loss
        commit_loss = F.mse_loss(z, z_q.detach())
        loss = self.beta * commit_loss

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()
        return z_q_st, codes.view(B, N), loss


# ───────────────────────── CNN encoder ────────────────────────

# Encoder transforms [B, L_s, C_s] to [B, L_l, D_l]
# L_l is called "n_latent_blocks"
# D_l is latent_dim
# We may use the same latent_dim across the whole autoencoder,
# since higher value doesn't make any sense.

class ConvSegmentEncoder(nn.Module):
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
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.r            = r

        # Learnable low-rank adapters only
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

    def forward(self, x):
        # x: [B, N, in_features]
        return (x @ self.A.T) @ self.B.T

# Taking 6 embedded sequences (using VQ's embedding), concatenating all of them
# then calculates the final sequence distribution
class TransformerSegmentDecoder(nn.Module):
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

        self.proj = LoRALinear(
            in_features  = 6 * n_latent_blocks * d_model,
            out_features = seg_len * n_codebooks * vocab_size,
            r = 8
        )

    def forward(
        self,
        z_prev_i, z_prev_v,
        z_curr_i, z_curr_v,
        z_next_i, z_next_v
    ):
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

#The VQ builds single-channel token vocabulary, with size = latent_vocab_size

class SegmentVQVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        seg_len: int,
        block_pairs: int = 4,
        latent_vocab_size: int = 512,
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
        self.vq_instru = VectorQuantizer(latent_vocab_size, latent_dim, beta)
        self.vq_vocal  = VectorQuantizer(latent_vocab_size, latent_dim, beta)

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
        use_vq: bool = True,
        zero_second_prob: float = 0.0,
        variance_loss_scale: float = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        B, T, C = tokens_curr.shape

        # encode three segments
        p_i, p_v = self.encoder(tokens_prev)
        c_i, c_v = self.encoder(tokens_curr)
        n_i, n_v = self.encoder(tokens_next)

        # Quantize normally
        p_i_q, p_c_q, vq_pi = self.vq_instru(p_i)
        c_i_q, c_c_q, vq_ci = self.vq_instru(c_i)
        n_i_q, n_c_q, vq_ni = self.vq_instru(n_i)

        p_v_q, _, vq_pv = self.vq_vocal(p_v)
        c_v_q, _, vq_cv = self.vq_vocal(c_v)
        n_v_q, _, vq_nv = self.vq_vocal(n_v)

        vq_loss = vq_pi + vq_ci + vq_ni + vq_pv + vq_cv + vq_nv

        logits = self.decoder(
            p_i_q, p_v_q,
            c_i_q, c_v_q,
            n_i_q, n_v_q
        )

        recon_loss = F.cross_entropy(
            logits.reshape(B * T * C, self.vocab_size),
            tokens_curr.reshape(B * T * C).long(),
            reduction="mean"
        )

        total_loss = recon_loss + vq_loss

        return total_loss, {
            "recon_loss": recon_loss.detach(),
            "vq_loss":    vq_loss.detach(),
            "total":      total_loss.detach(),
        }
    def train_ae(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        *,
        use_vq: bool = True,  # still accepts flag, but ignored here
        zero_second_prob: float = 0.0,
        variance_loss_scale: float = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        B, T, C = tokens_curr.shape

        # --- encode three segments directly ---
        p_i, p_v = self.encoder(tokens_prev)
        c_i, c_v = self.encoder(tokens_curr)
        n_i, n_v = self.encoder(tokens_next)

        # 🔁 skip VQ, use encoder output directly
        p_i_q = p_i
        c_i_q = c_i
        n_i_q = n_i

        p_v_q = p_v
        c_v_q = c_v
        n_v_q = n_v

        # --- decode ---
        logits = self.decoder(
            p_i_q, p_v_q,
            c_i_q, c_v_q,
            n_i_q, n_v_q
        )  # (B, T, C, V)

        # --- compute reconstruction loss only ---
        recon_loss = F.cross_entropy(
            logits.reshape(B * T * C, self.vocab_size),
            tokens_curr.reshape(B * T * C).long(),
            reduction="mean"
        )

        total_loss = recon_loss

        return total_loss, {
            "recon_loss": recon_loss.detach(),
            "vq_loss":    torch.tensor(0.0, device=logits.device),
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

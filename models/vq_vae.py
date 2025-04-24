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
 
# ────────────────────────── new constants ──────────────────────────
ENTROPY_WEIGHT = 0.25   # γ for codebook‐entropy regularizer
NCE_WEIGHT     = 0.10   # λ for latent predictive InfoNCE

def info_nce_loss(z: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    z: (B, N, D) latent sequence (e.g. pre‐VQ outputs)
    Computes InfoNCE between each block and its next neighbor,
    using all others in–batch as negatives.
    """
    B, N, D = z.shape
    # positives: t→t+1
    z0 = F.normalize(z[:, :-1, :].reshape(-1, D), dim=-1)  # (B*(N-1), D)
    z1 = F.normalize(z[:,  1:, :].reshape(-1, D), dim=-1)
    # similarity matrix
    sim = z0 @ z1.T                     # (B*(N-1), B*(N-1))
    labels = torch.arange(sim.size(0), device=z.device)
    return F.cross_entropy(sim / tau, labels)

# ───────────────────────── quantiser ──────────────────────────
class VectorQuantizer(nn.Module):
    """
    EMA-based VQ layer (drop-in replacement for the straight-through VectorQuantizer).
    """
    def __init__(self, num_codes: int, dim: int, beta: float = 0.1,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K = num_codes
        self.D = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        self.embed = nn.Embedding(self.K, self.D)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)

        # EMA buffers
        self.register_buffer('ema_cluster_size', torch.zeros(self.K))
        self.register_buffer('ema_w', self.embed.weight.data.clone())
    
    def code_usage_histogram(self, codes: torch.LongTensor) -> torch.Tensor:
        """
        codes: (B, N)
        returns: (K,) counts of each code index
        """
        hist = torch.bincount(
            codes.view(-1),
            minlength=self.K
        ).float()
        return hist

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
class _DWConvBlock(nn.Module):
    """
    Depth-wise separable 1-D conv (+ GLU) with residual & GroupNorm.
    """
    def __init__(self, chan_in: int, chan_mid: int):
        super().__init__()
        # depth-wise conv
        self.dw = nn.Conv1d(chan_in, chan_in, kernel_size=3,
                            padding=1, groups=chan_in, bias=False)
        # point-wise projection – doubled for GLU
        self.pw = nn.Conv1d(chan_in, 2 * chan_mid, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, chan_mid)          # channel-wise LayerNorm
        self.act  = nn.SiLU()                          # Swish
        # residual adapter in case of channel mismatch
        self.skip = (nn.Conv1d(chan_in, chan_mid, 1, bias=False)
                     if chan_in != chan_mid else nn.Identity())

    def forward(self, x):                              # x: (B, C, T)
        residual = self.skip(x)
        x = self.dw(x)
        x = self.pw(x)
        x, gate = x.chunk(2, dim=1)                    # GLU
        x = x * torch.sigmoid(gate)
        x = self.norm(x + residual)
        return self.act(x)

class LightweightSegmentEncoder(nn.Module):
    """
    Tiny CNN → latent grid  (B, N_blocks, 2·latent_dim)  then split streams.
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

        chan_in  = emb_dim * n_codebooks       # concat codebook token-embs
        chan_mid = latent_dim * 2              # modest width
        chan_out = latent_dim * 2              # final channels before split

        self.backbone = nn.Sequential(
            _DWConvBlock(chan_in,  chan_mid),
            _DWConvBlock(chan_mid, chan_mid),
            _DWConvBlock(chan_mid, chan_out),
        )

        # global average-pool to fixed latent grid (N_blocks)
        self.pool = nn.AdaptiveAvgPool1d(n_latent_blocks)

    # ───────────────────────────────────────────────────────────────────
    def forward(self, tokens: torch.Tensor):
        """
        tokens: (B, T_seg, C_codebooks)
        returns: (z_instr, z_vocal) -- each (B, N_blocks, latent_dim)
        """
        B, T, C = tokens.shape
        x = self.emb(tokens).view(B, T, -1).transpose(1, 2)  # (B, C′, T)
        x = self.backbone(x)                                 # (B, 2·D, T)
        x = self.pool(x).transpose(1, 2)                     # (B, N, 2·D)
        return x.chunk(2, dim=-1)                            # instru / vocal

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
        block_pairs: int = 4,
        num_codes: int = 512,
        emb_dim: int = 128,
        latent_dim: int = 128,
        beta: float = 0.05,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks

        self.encoder = LightweightSegmentEncoder(
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
        use_vq: bool = True,
        zero_second_prob: float = 0.0,
        variance_loss_scale: float = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        B, T, C = tokens_curr.shape

        # encode three segments
        p_i, p_v = self.encoder(tokens_prev)
        c_i, c_v = self.encoder(tokens_curr)
        n_i, n_v = self.encoder(tokens_next)

        # ───– InfoNCE on latent blocks (temporal predictive) ────
        nce_i = info_nce_loss(c_i)
        nce_v = info_nce_loss(c_v)
        nce_loss = nce_i + nce_v

    
        # Quantize normally (now capturing code indices)
        p_i_q, codes_pi, vq_pi = self.vq_instru(p_i)
        c_i_q, codes_ci, vq_ci = self.vq_instru(c_i)
        n_i_q, codes_ni, vq_ni = self.vq_instru(n_i)

        p_v_q, codes_pv, vq_pv = self.vq_vocal(p_v)
        c_v_q, codes_cv, vq_cv = self.vq_vocal(c_v)
        n_v_q, codes_nv, vq_nv = self.vq_vocal(n_v)

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

        # ───– Codebook‐usage entropy regulariser ────
        hist_i = (
            self.vq_instru.code_usage_histogram(codes_pi)
          + self.vq_instru.code_usage_histogram(codes_ci)
          + self.vq_instru.code_usage_histogram(codes_ni)
        )
        hist_v = (
            self.vq_vocal.code_usage_histogram(codes_pv)
          + self.vq_vocal.code_usage_histogram(codes_cv)
          + self.vq_vocal.code_usage_histogram(codes_nv)
        )
        code_hist = hist_i + hist_v                      # (K,)
        prob = code_hist / (code_hist.sum() + 1e-8)
        entropy_loss = -(prob * torch.log(prob + 1e-8)).sum()

        total_loss = (
              recon_loss
            + vq_loss
            + ENTROPY_WEIGHT * entropy_loss
            + NCE_WEIGHT     * nce_loss
        )
        return total_loss, {
            "recon_loss":    recon_loss.detach(),
            "vq_loss":       vq_loss.detach(),
            "entropy_loss":  entropy_loss.detach(),
            "nce_loss":      nce_loss.detach(),
            "total":         total_loss.detach(),
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

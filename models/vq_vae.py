from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Mixed (sinusoidal + learned scale) positional embedding.
    `d_model` must match the Transformer `d_model`.
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)           # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # not a parameter

        # learnable global scale
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.scale * self.pe[:T].unsqueeze(0)

# ───────────────────────── encoder ────────────────────────
class SegmentEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, seq_len: int, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len

        # First: local pattern extractor
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, latent_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        # Second: global temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4, dim_feedforward=latent_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(latent_dim, max_len=seq_len // 4 + 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        Output: (B, latent_dim)
        """
        x = x.transpose(1, 2)        # (B, D, T)
        x = self.conv(x)              # (B, latent_dim, T')
        x = x.transpose(1, 2)         # (B, T', latent_dim) for Transformer
        x = self.pos_enc(x)
        x = self.transformer(x)       # (B, T', latent_dim)
        x = x.transpose(1, 2)         # (B, latent_dim, T') for pooling
        x = self.pool(x).squeeze(-1)  # (B, latent_dim)
        x = self.fc(x)                # (B, latent_dim)
        return x



# ───────────────────── decoder ────────────────────
class SegmentDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, seq_len: int, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.pos_enc = PositionalEncoding(latent_dim, max_len=3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=latent_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # New layers for upsampling
        self.expand_proj = nn.Linear(latent_dim, latent_dim)
        self.temporal_expand = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),  # up x2
            nn.GELU(),
            nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1),  # up x2
            nn.GELU(),
            nn.Conv1d(latent_dim, output_dim, kernel_size=3, padding=1)  # final output channels
        )

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        z_prev, z_curr, z_next: (B, latent_dim)
        Output: (B, T, output_dim)
        """
        B = z_curr.size(0)

        # 1. Stack into sequence
        z = torch.stack([z_prev, z_curr, z_next], dim=1)  # (B, 3, latent_dim)

        # 2. Add positional encoding
        z = self.pos_enc(z)  # (B, 3, latent_dim)

        # 3. Transformer to mix
        z = self.transformer(z)  # (B, 3, latent_dim)

        # 4. Take the middle one (current latent after context)
        z_curr_ctx = z[:, 1, :]  # (B, latent_dim)

        # 5. Expand
        x = self.expand_proj(z_curr_ctx)  # (B, latent_dim)

        # 6. Treat as short sequence and upsample
        x = x.unsqueeze(-1)  # (B, latent_dim, 1)
        x = self.temporal_expand(x)  # (B, output_dim, T)

        x = x.transpose(1, 2)  # (B, T, output_dim)
        
        # 7. If necessary, cut or pad to match exactly seq_len
        if x.size(1) > self.seq_len:
            x = x[:, :self.seq_len, :]
        elif x.size(1) < self.seq_len:
            pad_size = self.seq_len - x.size(1)
            x = F.pad(x, (0, 0, 0, pad_size))  # pad along time dimension

        return x

# ───────────────────────── VQ-VAE wrapper ─────────────────────

#The VQ builds single-channel token vocabulary, with size = latent_vocab_size

class SegmentVQVAE(nn.Module):
    def __init__(
        self,
        input_dim:int=1024,
        latent_dim: int = 128,
        seq_len:int = 256
    ):
        super().__init__()
        self.encoder = SegmentEncoder(
            input_dim=input_dim, latent_dim=latent_dim, seq_len=seq_len
        )
        self.decoder = SegmentDecoder(
            output_dim=input_dim,latent_dim=latent_dim, seq_len=seq_len
        )

    # ─────────────────── forward ────────────────────
    def forward(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        use_vq: bool = True,
        zero_second_prob: float = 0.0,
        variance_loss_scale: float = 0.0,
        instrumental_mask = False,
        vocal_mask = False
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

        if instrumental_mask:
            p_v_q = torch.zeros_like(p_v_q)
            c_v_q = torch.zeros_like(c_v_q)
            n_v_q = torch.zeros_like(n_v_q)
            vq_loss += torch.mean(p_v)
            vq_loss += torch.mean(c_v)
            vq_loss += torch.mean(n_v)

        if vocal_mask:
            p_i_q = torch.zeros_like(p_i_q)
            c_i_q = torch.zeros_like(c_i_q)
            n_i_q = torch.zeros_like(n_i_q)
            vq_loss += torch.mean(p_i)
            vq_loss += torch.mean(c_i)
            vq_loss += torch.mean(n_i)
            

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
        tokens_next: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        tokens_prev, tokens_curr, tokens_next : (B, T, D)
        returns:
            loss: scalar loss
            metrics: dict of detached values
        """

        z_prev = self.encoder(tokens_prev)  # (B, latent_dim)
        z_curr = self.encoder(tokens_curr)  # (B, latent_dim)
        z_next = self.encoder(tokens_next)  # (B, latent_dim)

        z_all = torch.cat([z_prev, z_curr, z_next], dim=-1)  # (B, latent_dim * 3)

        recon = self.decoder(z_prev, z_curr, z_next)  # (B, T, D)

        # 5. Compute reconstruction loss
        recon_loss = F.l1_loss(recon, tokens_curr, reduction='mean')
        total_loss = recon_loss

        return total_loss, {
            "recon_loss": recon_loss.detach(),
            "total_loss": total_loss.detach()
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

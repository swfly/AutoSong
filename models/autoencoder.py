# models/vq_vae.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────── Attention Utility ───────────────────────

class CrossAttn2d(nn.Module):
    """
    2D ↔ Seq cross-attention (or self-attention if context_seq is None).
    - query_map: (B, C, H, W) → flatten to (B, H*W, C)
    - context_seq: (B, T, M)  → mel bins M, time steps T
    If mel_dim and max_context_len provided, projects M→C and adds pos.
    Otherwise does self-attn over the map.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        height: int,
        width: int,
        mel_dim: int | None = None,
        max_context_len: int | None = None,
        dropout: float = 0.1,
        resid_scale: float = 0.5,
    ):
        super().__init__()
        self.d_model   = d_model
        self.Lq        = height * width
        self.resid_sc  = resid_scale

        # positional for flattened map tokens
        self.pos_q     = nn.Parameter(torch.randn(1, self.Lq, d_model))

        # optional mel→token projection + positional
        if mel_dim is not None and max_context_len is not None:
            self.mel_proj = nn.Linear(mel_dim, d_model)
            self.pos_ctx  = nn.Parameter(torch.randn(1, max_context_len, d_model))
        else:
            self.mel_proj = None
            self.pos_ctx  = None

        # multihead attention
        self.attn = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

    def forward(
        self,
        query_map:   torch.Tensor,
        context_seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        query_map : (B, C, H, W)
        context_seq: (B, T, M) or None
        returns   : (B, C, H, W)
        """
        B, C, H, W = query_map.shape
        assert C == self.d_model and H * W == self.Lq

        # flatten query_map → (B, Lq, C)
        q = query_map.view(B, C, -1).permute(0, 2, 1)  # (B, Lq, C)
        q = q + self.pos_q[:, : self.Lq]              # add positional

        # prepare key/value
        if context_seq is None:
            k = v = q
        else:
            assert self.mel_proj is not None and self.pos_ctx is not None, \
                "Cross-attn requires mel_dim and max_context_len"
            B2, T, M = context_seq.shape
            assert B2 == B
            # project mel bins → d_model
            ctx = self.mel_proj(context_seq)          # (B, T, C)
            ctx = ctx + self.pos_ctx[:, :T]           # add mel-positional
            k = v = ctx                               # keys & values

        # cross- or self-attention
        out, _ = self.attn(q, k, v, need_weights=False)  # (B, Lq, C)

        # reshape back & residual
        out_map = out.permute(0, 2, 1).view(B, C, H, W)   # (B, C, H, W)
        return query_map + self.resid_sc * out_map


# ───────────────────────── CNN Blocks ─────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 5, stride=stride, padding=2)
        self.norm1    = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 5, padding=2)
        self.norm2    = nn.InstanceNorm2d(out_ch, affine=True)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, stride=stride)
                         if (in_ch!=out_ch or stride!=1) else nn.Identity())

    def forward(self, x):
        iden = self.shortcut(x)
        out  = F.gelu(self.norm1(self.conv1(x)))
        out  = F.gelu(self.norm2(self.conv2(out)))
        return F.gelu(out + iden)

class ResidualBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=(2,2)):
        super().__init__()
        self.scale    = scale_factor
        self.conv1    = nn.Conv2d(in_ch, out_ch, 5, padding=2)
        self.norm1    = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 5, padding=2)
        self.norm2    = nn.InstanceNorm2d(out_ch, affine=True)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch!=out_ch else nn.Identity())

    def forward(self, x):
        iden = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        iden = self.shortcut(iden)
        out  = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out  = F.gelu(self.norm1(self.conv1(out)))
        out  = F.gelu(self.norm2(self.conv2(out)))
        return F.gelu(out + iden)


# ───────────────────────── Encoder ────────────────────────────

class SegmentEncoder(nn.Module):
    """
    CNN encoder + cross-attention fusion.
    Input:  x ∈ ℝ^{B×T×M}   (e.g. 256×256 mel)
    Output: z ∈ ℝ^{B×C×H×W}
    """
    def __init__(
        self,
        input_size:      tuple[int,int],  # (T, M) mel dims
        output_size:     tuple[int,int],  # (H, W) latent spatial dims
        output_channels: int,             # C
        base_dim:        int = 32,
        max_dim:         int = 512,
    ):
        super().__init__()
        T, M = input_size
        H, W = output_size

        # 1) first conv
        self.initialize = nn.Conv2d(1, base_dim, 1)

        # 2) downsampling blocks
        self.blocks, ch = self._make_blocks(base_dim, input_size, output_size, base_dim, max_dim)

        # 3) project to C channels
        self.finalize = nn.Conv2d(ch, output_channels, 1)

        # 4) cross-attn fusion: mel_dim=M → C, map H×W
        self.fuse = CrossAttn2d(
            d_model         = output_channels,
            num_heads       = 1,
            height          = H,
            width           = W,
            mel_dim         = M,
            max_context_len = T,
            resid_scale     = 0.5,
        )

    def _make_blocks(self, in_ch, in_size, out_size, base_dim, max_dim):
        blocks = nn.ModuleList()
        curr_ch, (h, w) = in_ch, in_size
        H_out, W_out    = out_size
        while h > H_out or w > W_out:
            sh = 2 if h > H_out else 1
            sw = 2 if w > W_out else 1
            out_ch = min(base_dim*2, max_dim)
            blk    = ResidualBlock(curr_ch, out_ch, stride=(sh, sw))
            blocks.append(blk)
            curr_ch = out_ch
            h       = math.ceil(h / sh)
            w       = math.ceil(w / sw)
        return blocks, curr_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, M)
        returns z: (B, C, H, W)
        """
        B, T, M = x.shape
        y = x.unsqueeze(1)           # (B,1,T,M)
        y = self.initialize(y)
        for blk in self.blocks:
            y = blk(y)
        y = self.finalize(y)         # (B,C,H,W)
        return y
        return self.fuse(query_map=y, context_seq=x)


# ───────────────────────── Decoder ────────────────────────────

class SegmentDecoder(nn.Module):
    """
    Self-attention + up-conv decoder.
    Input:  z ∈ ℝ^{B×(3C)×H×W}
    Output: mel ∈ ℝ^{B×T×M}
    """
    def __init__(
        self,
        input_size:     tuple[int,int],  # (H, W)
        output_size:    tuple[int,int],  # (T, M)
        input_channels: int,             # 3*C
        base_dim:       int = 32,
        max_dim:        int = 512,
        attn_heads:     int = 4,
        dropout:      float = 0.1,
    ):
        super().__init__()
        H0, W0 = input_size
        T, M   = output_size

        # 1) project to base_dim
        self.initialize = nn.Conv2d(input_channels, base_dim, 1)

        # 2) self-attn on latent grid
        self.self_attn = CrossAttn2d(
            d_model         = base_dim,
            num_heads       = attn_heads,
            height          = H0,
            width           = W0,
            mel_dim         = None,
            max_context_len = None,
            resid_scale     = 0.5,
        )

        # 3) upsampling blocks
        self.blocks, ch = self._make_blocks(input_size, output_size, base_dim, max_dim)

        # 4) final conv to mel channel(s)
        self.finalize = nn.Conv2d(ch, 1, 1)

    def _make_blocks(self, in_size, out_size, base_dim, max_dim):
        blocks = nn.ModuleList()
        curr_ch = base_dim
        h, w    = in_size
        T, M    = out_size
        while h < T or w < M:
            sh = 2 if h < T else 1
            sw = 2 if w < M else 1
            out_ch = min(base_dim*2, max_dim)
            blk    = ResidualBlockUp(curr_ch, out_ch, scale_factor=(sh, sw))
            blocks.append(blk)
            curr_ch = out_ch
            h       = math.ceil(h * sh)
            w       = math.ceil(w * sw)
        return blocks, curr_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns mel: (B, T, M)
        """
        y = self.initialize(x)          # (B, base_dim, H, W)
        # y = self.self_attn(y, None)     # self-attn
        for blk in self.blocks:
            y = blk(y)
        mel = self.finalize(y).squeeze(1)  # (B, T, M)
        return mel


# ───────────────────────── VAE wrapper ───────────────────────

class SpectrogramDiscriminator(nn.Module):
    def __init__(self, input_channels=1, base_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, base_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(base_dim, base_dim*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(base_dim*2, base_dim*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(base_dim*4, base_dim*8, 4, 2, 1)
        self.fc    = nn.Linear(base_dim*8 * 16 * 16, 1)
        self.sig   = nn.Sigmoid()

    def forward(self, x):
        x    = x.unsqueeze(1)
        x    = F.leaky_relu(self.conv1(x), 0.2)
        x    = F.leaky_relu(self.conv2(x), 0.2)
        feat = x.clone()
        x    = F.leaky_relu(self.conv3(x), 0.2)
        x    = F.leaky_relu(self.conv4(x), 0.2)
        x    = x.flatten(1)
        return self.sig(self.fc(x)), feat


class SegmentAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim:           int = 256,
        latent_size: tuple[int,int] = (32,32),
        latent_channels:     int = 4,
        network_channel_base:int = 32,
        seq_len:             int = 256,
    ):
        super().__init__()
        C = latent_channels
        self.encoder = SegmentEncoder(
            input_size      = (seq_len, input_dim),
            output_size     = latent_size,
            output_channels = C,
            base_dim        = network_channel_base,
        )
        self.decoder = SegmentDecoder(
            input_size      = latent_size,
            output_size     = (seq_len, input_dim),
            input_channels  = C * 3,
            base_dim        = network_channel_base,
            attn_heads      = max(1, network_channel_base // 32),
        )

    def forward(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        mask_vocal:    bool = False,
        mask_inst:     bool = False,
    ):
        z_prev = self.encoder(tokens_prev) 
        z_curr = self.encoder(tokens_curr)
        z_next = self.encoder(tokens_next)

        C = z_prev.shape[1]
        z_prev_inst   = z_prev[:, 0:C//2, :, :]
        z_prev_vocal = z_prev[:, C//2:, :, :]

        z_curr_inst   = z_curr[:, 0:C//2, :, :]
        z_curr_vocal = z_curr[:, C//2:, :, :]

        z_next_inst   = z_next[:, 0:C//2, :, :]
        z_next_vocal = z_next[:, C//2:, :, :]

        if mask_vocal:
            z_prev_vocal *= 0.0
            z_curr_vocal *= 0.0
            z_next_vocal *= 0.0
        if mask_inst:
            z_prev_inst *= 0.0
            z_curr_inst *= 0.0
            z_next_inst *= 0.0

        z_prev = torch.cat([z_prev_inst, z_prev_vocal], dim=1)  # (B, C, H, W)
        z_curr = torch.cat([z_curr_inst, z_curr_vocal], dim=1)
        z_next = torch.cat([z_next_inst, z_next_vocal], dim=1)

        z = torch.concat([z_prev, z_curr, z_next], dim=1)  # (B, latent_dim * 3)
        recon = self.decoder(z)  # (B, T, D)

        # prior loss for stable latent distribution
        mu = z.mean(dim=0)
        std = z.std(dim=0)

        mean_loss = (mu ** 2).mean()         # mean → 0
        std_loss = ((std - 1) ** 2).mean()   # std → 1
        prior_loss = mean_loss + std_loss

        return recon, prior_loss

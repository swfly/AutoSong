# 📄 models/vq_vae.py

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, stride=stride, padding=2)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=2)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine = True)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        
        out += identity
        out = F.gelu(out)
        return out

class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2,2)):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5, stride=1, padding=2)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine = True)

        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut_conv = nn.Identity()

    def forward(self, x):
        identity = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        identity = self.shortcut_conv(identity)

        out = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.gelu(out)
        
        out += identity
        out = F.gelu(out)
        return out
    
# ───────────────────────── encoder ────────────────────────
class SegmentEncoder(nn.Module):
    def __init__(self, input_size, output_size, output_channels, base_dim=32, max_dim=512):
        super().__init__()
        
        # Initialize the input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.output_channels = output_channels
        
        self.initialize = nn.Conv2d(1, base_dim, kernel_size = 1)

        # Automatically calculate the number of residual blocks and their parameters
        self.blocks, out_dim = self.create_residual_blocks(input_size, output_size, base_dim, max_dim)
        
        # Final convolution to reduce channels to output_channels
        self.finalize = nn.Conv2d(out_dim, output_channels, kernel_size=1)

        self.fuse = CrossAttn2d(
            d_model         = output_channels,
            num_heads       = 1,
            height          = output_size[0],
            width           = output_size[1],
            mel_dim         = input_size[1],
            max_context_len = input_size[0],
            resid_scale     = 0.5,
        )

    def create_residual_blocks(self, input_size, output_size, base_dim, max_dim):
        # Calculate the necessary strides and channel expansions
        in_channels = base_dim  # Starting with 1 channel (grayscale input)
        current_size = input_size
        blocks = nn.ModuleList()
        
        # Keep doubling the channels and reducing the spatial size until we reach the target output size
        while current_size[0] > output_size[0] or current_size[1] > output_size[1]:
            stride_h = 2 if current_size[0] > output_size[0] else 1
            stride_w = 2 if current_size[1] > output_size[1] else 1
            
            out_channels = min(base_dim * 2, max_dim)  # Limit channel growth to max_dim
            block = ResidualBlock(in_channels, out_channels, stride=(stride_h, stride_w))
            blocks.append(block)
            
            # Update in_channels and current_size after applying the block
            in_channels = out_channels
            current_size = (math.ceil(current_size[0] / stride_h), math.ceil(current_size[1] / stride_w))
        
        return blocks, in_channels

    def forward(self, x):
        """
        x: (B, W, H)  -> output (B, W', H')
        """
        y = x.unsqueeze(1)  # (B, 1, W, H)
        y = self.initialize(y)

        for block in self.blocks:
            y = block(y)
        
        y = self.finalize(y)
        return self.fuse(query_map=y, context_seq=x)


# ───────────────────── decoder ────────────────────
class SegmentDecoder(nn.Module):
    def __init__(self, input_size, output_size, input_channels, output_channels, base_dim=32, max_dim=512):
        super().__init__()
        
        # Initialize the input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.output_channels = output_channels
        self.input_channels= input_channels
        
        self.initialize = nn.Conv2d(input_channels, base_dim, kernel_size = 1)

        self.self_attn = CrossAttn2d(
            d_model         = base_dim,
            num_heads       = 4,
            height          = input_size[0],
            width           = input_size[1],
            mel_dim         = None,
            max_context_len = None,
            resid_scale     = 0.5,
        )

        # Automatically calculate the number of residual blocks and their parameters
        self.blocks, out_channel = self.create_residual_blocks(input_size, output_size, base_dim, max_dim)
        
        # Final convolution to reduce channels to output_channels
        self.finalize = nn.Conv2d(out_channel, output_channels, kernel_size=1)

    def create_residual_blocks(self, input_size, output_size, base_dim, max_dim):
        # Calculate the necessary strides and channel expansions
        in_channels = base_dim
        current_size = input_size
        blocks = nn.ModuleList()
        
        # Keep doubling the channels and increasing the spatial size until we reach the target output size
        while current_size[0] < output_size[0] or current_size[1] < output_size[1]:
            stride_h = 2 if current_size[0] < output_size[0] else 1
            stride_w = 2 if current_size[1] < output_size[1] else 1
            
            out_channels = min(base_dim * 2, max_dim)  # Limit channel growth to max_dim
            block = ResidualBlockUp(in_channels, out_channels, scale_factor=(stride_h, stride_w))
            blocks.append(block)
            
            # Update in_channels and current_size after applying the block
            in_channels = out_channels
            current_size = (math.ceil(current_size[0] * stride_h), math.ceil(current_size[1] * stride_w))
        
        return blocks, in_channels

    def forward(self, x):
        """
        x: (B, C, W, H)  -> output (B, 1, W', H')
        """
        x = self.initialize(x)
        x = self.self_attn(x, None)     # self-attn

        for block in self.blocks:
            x = block(x)
            
        x = self.finalize(x)
        return x.squeeze(1)

    
# ───────────────────────── VAE wrapper ─────────────────────
class SpectrogramDiscriminator(nn.Module):
    def __init__(self, input_channels=1, base_dim=32, patch_size = 256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, base_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_dim, base_dim*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_dim*2, base_dim*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_dim*4, base_dim*8, kernel_size=4, stride=2, padding=1)
        
        size = int(patch_size>>4)
        self.fc = nn.Linear(base_dim*8 * size * size, 1)  # Adjust based on the size of the input
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (B, 1, W, H)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        y = x
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, -1)
        x = self.fc(x)
        return self.sigmoid(x), y  # Output probability for real or fake

class SegmentAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim:int=128,
        latent_size = (16,16),
        latent_channels = 2,
        network_channel_base: int = 16,
        seq_len:int = 256
    ):
        super().__init__()

        feature_channels = latent_channels

        self.encoder = SegmentEncoder(
            input_size=(seq_len, input_dim), output_size=latent_size, output_channels=feature_channels, base_dim = network_channel_base
        )
        self.decoder = SegmentDecoder(
            output_size=(seq_len, input_dim), input_size=latent_size, 
            input_channels=feature_channels*3, output_channels=1, base_dim = network_channel_base + network_channel_base//2
        )

    # ─────────────────── forward ────────────────────:
    def forward(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
        mask_vocal = False,
        mask_inst = False
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

        track_loss = torch.zeros((1)).to(z_curr.device)  # Initialize track loss

        if mask_vocal:
            track_loss += (z_prev_vocal**2).mean() + (z_curr_vocal**2).mean() + (z_next_vocal**2).mean()
            z_prev_vocal = z_prev_vocal * 0.0
            z_curr_vocal = z_curr_vocal * 0.0
            z_next_vocal = z_next_vocal * 0.0

        if mask_inst:
            track_loss += (z_prev_inst**2).mean() + (z_curr_inst**2).mean() + (z_next_inst**2).mean()
            z_prev_inst = z_prev_inst * 0.0
            z_curr_inst = z_curr_inst * 0.0
            z_next_inst = z_next_inst * 0.0
            

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
        prior_loss = mean_loss + std_loss + track_loss

        return recon, prior_loss


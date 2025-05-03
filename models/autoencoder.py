# 📄 models/vq_vae.py

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x.unsqueeze(1)  # (B, 1, W, H)
        x = self.initialize(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.finalize(x)
        return x


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

        for block in self.blocks:
            x = block(x)
            
        x = self.finalize(x)
        return x.squeeze(1)

    
# ───────────────────────── VAE wrapper ─────────────────────
class SpectrogramDiscriminator(nn.Module):
    def __init__(self, input_channels=1, base_dim=32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, base_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_dim, base_dim*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_dim*2, base_dim*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_dim*4, base_dim*8, kernel_size=4, stride=2, padding=1)
        
        self.fc = nn.Linear(base_dim*8 * 16 * 16, 1)  # Adjust based on the size of the input
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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


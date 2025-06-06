# 📄 models/vq_vae.py

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gumbel_softmax_quantizer import GumbelSoftmaxQuantizer


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

# ───────────────────────── VQ ────────────────────────
class VectorQuantizerEMA(nn.Module):
    """
    Compress EACH (H,W) map of every channel into ONE codebook vector,
    then broadcast it back to (H,W).

    z            : (B, C, H, W)
    q (returned) : (B, C, H, W)   (straight-through)
    indices      : (B, C)         discrete token IDs
    """
    def __init__(self, num_embeddings, patch_hw, num_channels, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.K = num_embeddings
        self.D = patch_hw
        self.C = num_channels
        self.beta = commitment_cost
        self.decay = decay
        self.eps = eps

        # Channel-wise codebooks
        embed = torch.randn(self.C, self.K, self.D)  # (C, K, H*W)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(self.C, self.K))
        self.register_buffer("ema_embed_sum", embed.clone())

    def forward(self, z):
        B, C, H, W = z.shape
        assert C == self.C and H * W == self.D

        z_flat = z.view(B, C, -1)  # (B, C, H*W)
        z_flat = z_flat.transpose(0, 1)  # (C, B, D)

        all_indices = []
        all_quantized = []

        for c in range(self.C):
            x = z_flat[c]  # (B, D)
            embed = self.embedding[c]  # (K, D)

            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ embed.t()
                + embed.pow(2).sum(1)
            )
            indices = dist.argmin(1)  # (B,)
            quantized = embed[indices]  # (B, D)

            if self.training:
                onehot = F.one_hot(indices, self.K).type_as(x)
                n_i = onehot.sum(0).detach()
                e_i = onehot.t() @ x.detach()

                self.ema_cluster_size[c].mul_(self.decay).add_(n_i, alpha=1 - self.decay)
                self.ema_embed_sum[c].mul_(self.decay).add_(e_i, alpha=1 - self.decay)

                n_tot = self.ema_cluster_size[c].sum()
                smoothed_n = (self.ema_cluster_size[c] + self.eps) / (
                    n_tot + self.K * self.eps
                ) * n_tot
                self.embedding[c].copy_(self.ema_embed_sum[c] / smoothed_n.unsqueeze(1))

            all_indices.append(indices)
            all_quantized.append(quantized)

        quantized = torch.stack(all_quantized, dim=1)  # (B, C, D)
        quantized = quantized.view(B, C, H, W)
        indices = torch.stack(all_indices, dim=1)  # (B, C)

        commit_loss = F.mse_loss(quantized.detach(), z)
        q_st = z + (quantized - z).detach()

        return q_st, self.beta * commit_loss, indices
    
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

class SegmentVAE(nn.Module):
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

        self.vq_inst = VectorQuantizerEMA(
        num_embeddings = 1024,
        patch_hw       = latent_size[0] * latent_size[1],   # 256 in this example
        num_channels = feature_channels // 2,
        # hard=False
        )

        self.vq_vocal = VectorQuantizerEMA(
        num_embeddings = 1024,
        patch_hw       = latent_size[0] * latent_size[1],   # 256 in this example
        num_channels = feature_channels // 2,
        # hard = False
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

        ae_only = False
        if ae_only:
            z_prev_q_inst = z_prev_inst
            z_curr_q_inst = z_curr_inst
            z_next_q_inst = z_next_inst

            z_prev_q_vocal = z_prev_vocal
            z_curr_q_vocal = z_curr_vocal
            z_next_q_vocal = z_next_vocal
            vq_loss = 0.0
        else:
            z_prev_q_inst, prev_vq_loss, token_prev = self.vq_inst(z_prev_inst)
            z_curr_q_inst, curr_vq_loss, token_curr = self.vq_inst(z_curr_inst)
            z_next_q_inst, next_vq_loss, token_next = self.vq_inst(z_next_inst)

            vq_loss = prev_vq_loss+curr_vq_loss+next_vq_loss

            z_prev_q_vocal, prev_vq_loss, token_prev = self.vq_vocal(z_prev_vocal)
            z_curr_q_vocal, curr_vq_loss, token_curr = self.vq_vocal(z_curr_vocal)
            z_next_q_vocal, next_vq_loss, token_next = self.vq_vocal(z_next_vocal)

            vq_loss += prev_vq_loss+curr_vq_loss+next_vq_loss

        z_prev_q = torch.cat([z_prev_q_inst, z_prev_q_vocal], dim=1)  # (B, C, H, W)
        z_curr_q = torch.cat([z_curr_q_inst, z_curr_q_vocal], dim=1)
        z_next_q = torch.cat([z_next_q_inst, z_next_q_vocal], dim=1)

        z = torch.concat([z_prev_q, z_curr_q, z_next_q], dim=1)  # (B, latent_dim * 3)

        recon = self.decoder(z)  # (B, T, D)
        return recon, vq_loss
    
    def train_ae(
        self,
        tokens_prev: torch.Tensor,
        tokens_curr: torch.Tensor,
        tokens_next: torch.Tensor,
    ):
        z_prev = self.encoder(tokens_prev) 
        z_curr = self.encoder(tokens_curr)
        z_next = self.encoder(tokens_next)

        z = torch.concat([z_prev, z_curr, z_next], dim=1)  # (B, latent_dim * 3)
        recon = self.decoder(z)  # (B, T, D)

        # 5. Compute reconstruction loss
        recon_loss = F.l1_loss(recon, tokens_curr, reduction='mean')
        # perceptual_loss = self.perceptual_loss_fn(recon, tokens_curr)

        total_loss = recon_loss

        return total_loss, {
            "recon_loss": recon_loss.detach(),
            "total_loss": total_loss.detach()
        }


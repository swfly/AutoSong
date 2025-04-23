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

class SegmentEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        emb_dim: int,
        latent_dim: int,
        n_latent_blocks: int  # number of (z_instru, z_vocal) pairs per segment
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.n_latent_blocks = n_latent_blocks

        self.token_emb = nn.Embedding(vocab_size, emb_dim)

        input_dim = emb_dim * n_codebooks
        hidden_dim = 2 * latent_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 2 * latent_dim, kernel_size=1),  # outputs instru + vocal
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (B, T=256, C)
        returns:
          z_instru: (B, N, D)
          z_vocal : (B, N, D)
        """
        B, T, C = tokens.shape
        x = self.token_emb(tokens)                        # (B, T, C, emb)
        x = x.view(B, T, -1).permute(0, 2, 1)             # (B, emb*C, T)
        x = self.encoder_cnn(x)                           # (B, 2*D, T)

        # Pool to fixed N blocks (latent units)
        x = F.adaptive_avg_pool1d(x, self.n_latent_blocks)  # (B, 2*D, N)
        x = x.permute(0, 2, 1)                             # (B, N, 2*D)

        z_instru, z_vocal = x.chunk(2, dim=-1)            # → each (B, N, D)
        return z_instru, z_vocal
class SegmentDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        latent_dim: int,
        n_latent_blocks: int,
        seg_len: int
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_codebooks = n_codebooks
        self.seg_len = seg_len

        # Total input dim: 6 sets of (N, D)
        self.input_dim = 6 * latent_dim * n_latent_blocks
        hidden_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, vocab_size * n_codebooks, kernel_size=1),
        )

    def forward(
        self,
        z_prev_instru, z_prev_vocal,
        z_curr_instru, z_curr_vocal,
        z_next_instru, z_next_vocal
    ) -> torch.Tensor:
        """
        Each input: (B, N, D)
        Output logits: (B, T, C, V)
        """
        z_all = torch.cat([
            z_prev_instru, z_prev_vocal,
            z_curr_instru, z_curr_vocal,
            z_next_instru, z_next_vocal
        ], dim=1)  # → (B, 6N, D)

        z = z_all.view(z_all.size(0), -1)       # → (B, 6ND)
        h = self.fc(z)                          # → (B, hidden)
        h = h.unsqueeze(-1).expand(-1, -1, self.seg_len)  # (B, hidden, T)
        out = self.decoder_conv(h)              # (B, VC, T)

        B, VC, T = out.shape
        out = out.view(B, self.n_codebooks, self.vocab_size, T)
        out = out.permute(0, 3, 1, 2)           # (B, T, C, V)
        return out

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
        block_pairs:int = 2,
        num_codes: int = 1024,
        emb_dim: int = 128,
        latent_dim: int = 256,
        beta: float = 0.1,
    ):
        super().__init__()
        self.seg_len     = seg_len
        self.n_codebooks = n_codebooks
        self.vocab_size  = vocab_size
        self.pair = block_pairs

        self.encoder = SegmentEncoder(vocab_size=vocab_size, n_codebooks=n_codebooks,
                            emb_dim=emb_dim, latent_dim=latent_dim, n_latent_blocks=block_pairs)
        self.decoder = SegmentDecoder(vocab_size=vocab_size, n_codebooks=n_codebooks,
                            latent_dim=latent_dim, n_latent_blocks=block_pairs, seg_len=seg_len)


        # classifier: 1×1 conv to vocab logits
        self.classifier = nn.Conv1d(emb_dim * n_codebooks, vocab_size * n_codebooks, kernel_size=1)


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
        B, T, C = tokens_curr.shape

        # --- encode all three segments ---
        z_prev_instru, z_prev_vocal = self.encoder(tokens_prev)   # (B, N, D)
        z_curr_instru, z_curr_vocal = self.encoder(tokens_curr)
        z_next_instru, z_next_vocal = self.encoder(tokens_next)

        # --- maybe zero z_curr_vocal ---
        # if self.training and zero_second_prob > 0.0:
        #     if torch.rand(1).item() < zero_second_prob:
        #         z_curr_vocal = torch.zeros_like(z_curr_vocal)

        # --- concatenate latent blocks from all three segments ---
        logits = self.decoder(
            z_prev_instru, z_prev_vocal,
            z_curr_instru, z_curr_vocal,
            z_next_instru, z_next_vocal
        )


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
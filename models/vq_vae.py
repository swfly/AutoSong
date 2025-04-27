from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def assert_finite(t, name):
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"{name} contains NaN/Inf")
# -----------------------------------------------------------------------------
#  LoRA utility (optional – not used inside the AE, but provided for extension)
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-rank adaptation layer:  ΔW = B·A  (rank *r*)."""

    def __init__(self, in_features: int, out_features: int, r: int = 4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, in]
        return x @ (self.A.t() @ self.B.t())
class gMLPBlock(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion_factor * d_model)
        self.sgu = nn.Sequential(
            nn.Linear(expansion_factor * d_model, expansion_factor * d_model),
            nn.GELU()
        )
        self.fc2 = nn.Linear(expansion_factor * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.sgu(x)
        x = self.fc2(x)
        return x + residual
# -----------------------------------------------------------------------------
#  Encoder
# -----------------------------------------------------------------------------
class DiscreteEncoder(nn.Module):
    """Map *EnCodec* tokens → distribution over latent codes of length *L*.

    Input : tokens          [B, T, C]
    Output: latent logits   [B, L, C, latent_vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        n_codebooks: int,
        seq_len: int,                  # maximum input T
        latent_seq_len: int,           # desired latent length L
        latent_vocab_size: int = 512,
        embed_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_seq_len = latent_seq_len
        self.latent_vocab_size = latent_vocab_size
        self.n_codebooks = n_codebooks
        D_e, D_l = embed_dim, latent_dim

        # token / channel / position embeddings
        self.token_emb   = nn.Embedding(vocab_size, D_e)
        self.channel_emb = nn.Embedding(n_codebooks, D_e)
        self.pos_emb     = nn.Parameter(torch.randn(1, seq_len, D_e) * 0.02)
        nn.init.uniform_(self.token_emb.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.channel_emb.weight, a=-1.0, b=1.0)

        self.in_proj = nn.Linear(D_e, D_l) if D_e != D_l else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_l,
            nhead=num_heads,
            dim_feedforward=4 * D_l,
            dropout=dropout,
            batch_first=True,
        )
        self.tfm = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool_proj = nn.Linear(seq_len * n_codebooks, latent_seq_len)
        self.out_proj = nn.Linear(D_l, latent_vocab_size)

    # ---------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [B,T,C]
        B, T, C = tokens.shape
        assert C == self.n_codebooks, "C does not match n_codebooks"

        # Embedding + positional & channel encodings ––> [B, T, C, D_e]
        x = self.token_emb(tokens.view(B, T * C)).view(B, T, C, -1)
        ch = self.channel_emb(torch.arange(C, device=tokens.device)).view(1, 1, C, -1)
        pos = self.pos_emb[:, :T, :].unsqueeze(2)
        x = x + ch + pos

        # [B, T*C, D_e] → project → Transformer
        x = self.in_proj(x.view(B, T * C, -1))      # [B, T*C, D_l]
        x = self.tfm(x)                           # [B, D_l, T*C]
        
        # project [T*C, D_l] to [L_l, D_l] via pooling
        # x: [B, T*C, D_l]
        x = x.permute(0, 2, 1)  # [B, D_l, T*C]
        # Downsample sequence: T*C -> L_l
        x = self.pool_proj(x)                      # [B, D_l, L]
        x = x.permute(0, 2, 1) 

        # Project to distribution
        logits = self.out_proj(x)            # [B, L_l, V_l]
        return logits

# -----------------------------------------------------------------------------
#  Decoder
# -----------------------------------------------------------------------------
class DiscreteDecoder(nn.Module):
    """Reconstruct original tokens from latent codes.

    latent tokens:  [B, L, C] or embedded [B, L, C, D_e]
    output logits : [B, T, C, vocab_size]
    """

    def __init__(
        self,
        latent_vocab_size: int,
        latent_len: int,
        output_token_vocab_size: int,
        n_codebooks: int,
        target_seq_len: int,   # original T (for up-sampling)
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.decode_segment_len = target_seq_len
        self.n_codebooks = n_codebooks
        self.embed_dim = embed_dim
        self.latent_emb  = nn.Embedding(latent_vocab_size, embed_dim)
        self.pos_emb_lat = nn.Parameter(torch.randn(1, latent_len, embed_dim) * 0.02)
        self.channel_emb = nn.Embedding(n_codebooks, embed_dim)
        nn.init.uniform_(self.latent_emb.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.channel_emb.weight, a=-1.0, b=1.0)
        
        D_l = embed_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_l,
            nhead=num_heads,
            dim_feedforward=4 * D_l,
            dropout=dropout,
            batch_first=True,
        )
        self.tfm = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, output_token_vocab_size)

    def straight_through_softmax(self, logits, tau=1.0):
        # Soft sample
        soft = F.softmax(logits / tau, dim=-1)
        # Hard sample
        index = soft.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # Straight-through
        return hard + soft - hard.detach()
    # full discrete path -------------------------------------------------
    def forward_soft_embedding(self, z_logits): #[B, L_l, V_l] token sequence, no channel
        B, L_l, V = z_logits.shape
        D = self.embed_dim
        C = self.n_codebooks
        T = self.decode_segment_len
        # 1. Soft-Embed latent tokens
        z_soft = self.straight_through_softmax(z_logits)
        z_emb = torch.einsum('blv,vd->bld', z_soft, self.latent_emb.weight)  # [B, L_l, D]
        # # 2. Add latent positional embedding
        pos_lat = self.pos_emb_lat[:, :L_l, :]  # [1, L_l, D]
        z_emb = z_emb + pos_lat  # [B, L_l, D]
        z_emb = z_emb.unsqueeze(2).expand(-1, -1, C, -1)  # [B, L_l, C, D]
        # 4. Add channel embedding
        ch_emb = self.channel_emb(torch.arange(C, device=z_logits.device)).view(1, 1, C, D)
        z_emb = z_emb + ch_emb  # [B, L_l, C, D]
        # 5. Upsample L_l → T
        z_emb = z_emb.permute(0, 2, 3, 1)  # [B, C, D, L_l]
        z_emb = z_emb.reshape(B * C, D, L_l)  # [B*C, D, L_l]
        z_emb = F.interpolate(z_emb, size=T, mode="nearest")  # [B*C, D, T]
        z_emb = z_emb.view(B, C, D, T).permute(0, 3, 1, 2)  # [B, T, C, D]
        z_emb = z_emb.reshape(B, T * C, D)  # [B, T*C, D]
        x = z_emb
        x = self.tfm(x)         # [B, D, T*C]
        # 9. Reshape back
        x = x.view(B, T, C, D)  # [B, T, C, D]

        # 10. Output projection per codebook
        logits = self.out_proj(x)  # [B, T, C, vocab_size]
        return logits

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:  # [B,L_l]
        B, L_l = z_tokens.shape
        D = self.embed_dim
        C = self.n_codebooks
        T = self.decode_segment_len

        # 1. Embed latent tokens
        z_emb = self.latent_emb(z_tokens)  # [B, L_l, D]
        # 2. Add latent positional embedding
        pos_lat = self.pos_emb_lat[:, :L_l, :]  # [1, L_l, D]
        z_emb = z_emb + pos_lat  # [B, L_l, D]
        z_emb = z_emb.unsqueeze(2).expand(-1, -1, C, -1)  # [B, L_l, C, D]
        # 4. Add channel embedding
        ch_emb = self.channel_emb(torch.arange(C, device=z_tokens.device)).view(1, 1, C, D)
        z_emb = z_emb + ch_emb  # [B, L_l, C, D]
        # 5. Upsample L_l → T
        z_emb = z_emb.permute(0, 2, 3, 1)  # [B, C, D, L_l]
        z_emb = z_emb.reshape(B * C, D, L_l)  # [B*C, D, L_l]
        z_emb = F.interpolate(z_emb, size=T, mode="nearest")  # [B*C, D, T]
        z_emb = z_emb.view(B, C, D, T).permute(0, 3, 1, 2)  # [B, T, C, D]
        z_emb = z_emb.reshape(B, T * C, D)  # [B, T*C, D]
        x = z_emb
        x = self.tfm(x)         # [B, D, T*C]
        # 9. Reshape back
        x = x.view(B, T, C, D)  # [B, T, C, D]

        # 10. Output projection per codebook
        logits = self.out_proj(x)  # [B, T, C, vocab_size]
        return logits

# -----------------------------------------------------------------------------
#  Auto-encoder wrapper
# -----------------------------------------------------------------------------
class TransformerAutoencoder(nn.Module):
    """Encode → sample → decode."""

    def __init__(
        self,
        *,
        input_token_vocab_size: int,
        n_codebooks: int,
        segment_length: int,          # maximum input T
        latent_seq_len: int,          # compressed length L
        latent_vocab_size: int = 512,
        embed_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # encoder
        self.encoder = DiscreteEncoder(
            vocab_size=input_token_vocab_size,
            n_codebooks=n_codebooks,
            seq_len=segment_length,
            latent_seq_len=latent_seq_len,
            latent_vocab_size=latent_vocab_size,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # decoder
        self.decoder = DiscreteDecoder(
            latent_vocab_size=latent_vocab_size,
            latent_len=latent_seq_len,
            output_token_vocab_size=input_token_vocab_size,
            n_codebooks=n_codebooks,
            target_seq_len=segment_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
    # ------------------------------------------------------------------
    def _gumbel_soft(self, logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        g = -torch.empty_like(logits).exponential_().log() + 1e-8  # Gumbel(0,1)
        return F.softmax((logits + g) / tau, dim=-1)
    def gumbel_sample(self,logits, tau=1.0):
        eps = 1e-20
        noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        return F.softmax((logits + noise) / tau, dim=-1)
    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> dict:
        """Full forward pass.

        tokens: [B, T, C]
        returns: {loss, enc_logits, dec_logits, z_tokens}
        """
        enc_logits = self.encoder(tokens)                    # [B,L_l,V_l]      
        # z_tokens = enc_logits.argmax(dim=-1)                  # [B,L_l]
        # dec_logits = self.decoder(z_tokens)
        dec_logits = self.decoder.forward_soft_embedding(enc_logits)    # [B,T,C,V]
        # loss = F.cross_entropy(dec_logits.view(-1, dec_logits.size(-1)), tokens.view(-1))
        loss = F.cross_entropy(
            dec_logits.reshape(-1, dec_logits.size(-1)),  # [B*T*C, vocab_size]
            tokens.reshape(-1)                            # [B*T*C]
        )
        return {
            "loss": loss
        }
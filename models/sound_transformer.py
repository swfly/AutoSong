# ──────────────────────────────── Composer ✕ Autoregressor ────────────────────────────────

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vocabulary as vocab_mod

# ──────────────────────────────── Composer ────────────────────────────────
class Composer(nn.Module):
    """Level‑1 network: lyrics → 1‑s block embeddings + heads."""

    def __init__(
        self,
        block_num: int = 180,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        max_text_len: int = 512,
        phoneme_len: int = 32,
        n_roles: int = 8,
        n_dur_slots: int = 16,
    ) -> None:
        super().__init__()
        self.block_num   = block_num
        self.d_model     = d_model
        self.phoneme_len = phoneme_len

        # vocab / token embedding
        self.vocab = vocab_mod.generate_pinyin_vocab()
        self.text_emb     = nn.Embedding(len(self.vocab), d_model)
        self.pos_emb_text = nn.Parameter(torch.randn(1, max_text_len, d_model) * (1.0 / d_model ** 0.5))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(enc_layer, n_layers)

        # block queries ---------------------------------------------------
        self.block_queries = nn.Parameter(torch.randn(1, block_num, d_model) * (1.0 / d_model ** 0.5))
        self.block_pos    = nn.Embedding(block_num, d_model)
        self.dur_emb      = nn.Embedding(n_dur_slots, d_model)

        # two continuous scalar → bias projections
        self.rel_track_lin = nn.Linear(1, d_model)  # position over full 3‑min track
        self.rel_sent_lin  = nn.Linear(1, d_model)  # position inside current sentence

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.block_decoder = nn.TransformerDecoder(dec_layer, n_layers)

        # heads -----------------------------------------------------------
        self.phoneme_head = nn.Linear(d_model, phoneme_len * len(self.vocab))
        self.blank_head   = nn.Linear(d_model, 1)
        self.role_head    = nn.Linear(d_model, n_roles)
        self.role_emb     = nn.Parameter(torch.randn(n_roles, d_model))
        self.blank_vec    = nn.Parameter(torch.randn(1, 1, d_model))

    # ------------------------------------------------------------------
    def forward(
        self,
        text_ids: torch.Tensor,                     # (B, T_txt)
        duration_ids: Optional[torch.Tensor] = None,    # (B, S)
        sent_rel_pos: Optional[torch.Tensor] = None,    # (B, S) floats 0‑1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (phoneme_logits, blank_logits, role_weights, block_mem)."""

        device = text_ids.device
        B, T = text_ids.shape
        S = self.block_num

        # (1) Text encoding ---------------------------------------------
        mem = self.text_encoder(self.text_emb(text_ids) + self.pos_emb_text[:, :T])

        # (2) Build queries --------------------------------------------
        q = self.block_queries.expand(B, -1, -1)
        q = q + self.block_pos(torch.arange(S, device=device))[None]

        # track‑relative scalar bias (0‑1 at block centres) --------------
        rel_track = torch.linspace(0, 1, steps=S, device=device).view(1, S, 1)
        q = q + self.rel_track_lin(rel_track)

        if duration_ids is not None:
            q = q + self.dur_emb(duration_ids)

        if sent_rel_pos is not None:
            q = q + self.rel_sent_lin(sent_rel_pos.unsqueeze(-1))

        blocks = self.block_decoder(q, mem)  # (B,S,d)

        # (3) Heads ------------------------------------------------------
        ph_logits = self.phoneme_head(blocks).view(B, S, self.phoneme_len, len(self.vocab))
        blank_logits = self.blank_head(blocks).squeeze(-1)

        role_logits  = self.role_head(blocks)
        role_weights = torch.softmax(role_logits, -1)
        role_bias    = role_weights @ self.role_emb
        blank_bias   = torch.sigmoid(blank_logits).unsqueeze(-1) * self.blank_vec

        block_mem = blocks + role_bias + blank_bias
        return ph_logits, blank_logits, role_weights, block_mem

# ──────────────────────────────── Helpers ────────────────────────────────

def _build_causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)

def _build_local_causal_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        m[i, max(0, i - window): i + 1] = 0.0
    return m

# ───────────────────────────── Autoregressor ─────────────────────────────
class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is not None:
            y = self.ln1(x)
            x = x + self.self_attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        y = self.ln2(x)
        x = x + self.cross_attn(y, mem, mem, need_weights=False)[0]
        y = self.ln3(x)
        x = x + self.mlp(y)
        return x

class Autoregressor(nn.Module):
    """Level‑2 autoregressive Transformer operating on latent patches."""

    def __init__(
        self,
        in_channels: int = 4,
        patch_hw: Tuple[int, int] = (32, 32),
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 180,
        window: int = 8,
    ) -> None:
        super().__init__()
        H, W = patch_hw
        self.C, self.H, self.W = in_channels, H, W

        self.chan_emb = nn.Embedding(in_channels, H * W)
        self.time_pos = nn.Parameter(torch.randn(1, max_seq_len, H * W) * (1.0 / (H * W) ** 0.5))

        self.token_proj = nn.Sequential(
            nn.Linear(in_channels * H * W, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.blocks_global = nn.ModuleList(_DecoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.blocks_local  = nn.ModuleList(_DecoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.fuse          = _DecoderLayer(d_model, n_heads)

        self.recon_proj = nn.Sequential(
            nn.Linear(d_model, 4 * in_channels * H * W), nn.GELU(),
            nn.Linear(4 * in_channels * H * W, in_channels * H * W),
        )
        self.delta_scale = nn.Parameter(torch.full((in_channels,), 0.1))

        # pre‑built masks (on CPU)
        self.register_buffer("_mask_global", _build_causal_mask(max_seq_len, torch.device("cpu")), persistent=False)
        self.register_buffer("_mask_local",  _build_local_causal_mask(max_seq_len, window, torch.device("cpu")), persistent=False)

    # ------------------------------------------------------------------
    def forward(self, latents: torch.Tensor, block_mem: torch.Tensor) -> torch.Tensor:
        """latents (B,S,C,H,W) → reconstructed latents same shape"""
        B, S, C, H, W = latents.shape
        assert (C, H, W) == (self.C, self.H, self.W)
        device = latents.device

        x = latents.view(B, S, C, H * W)
        # add channel & time positional
        x = x + self.chan_emb.weight[:C].view(1, 1, C, -1)  # safe slice
        x = x + self.time_pos[:, :S].view(1, S, 1, -1)
        tokens = self.token_proj(x.view(B, S, -1))          # (B,S,d)

        mg = self._mask_global[:S, :S].to(device)
        ml = self._mask_local[:S, :S].to(device)

        tg, tl = tokens, tokens
        for blk in self.blocks_global:
            tg = blk(tg, block_mem, mg)
        for blk in self.blocks_local:
            tl = blk(tl, block_mem, ml)

        fused = tl + self.fuse(tl, tg, mg)  # fuse keeps causal mask
        out   = self.recon_proj(fused).view(B, S, C, H, W)
        return latents + torch.relu(self.delta_scale.view(1, 1, -1, 1, 1)) * out

# ─────────────────────────────── End‑to‑End wrapper ───────────────────────────────
class EndToEndNetwork(nn.Module):
    def __init__(self, composer_cfg: dict, autoregressor_cfg: dict):
        super().__init__()
        self.composer = Composer(**composer_cfg)
        self.decoder  = Autoregressor(**autoregressor_cfg)

    # ------------------------------------------------------------------
    def forward(
        self,
        text_ids: torch.Tensor,
        phoneme_tgt: torch.Tensor,
        silence_mask: torch.Tensor,
        latents: torch.Tensor,
        duration_ids: Optional[torch.Tensor] = None,
        loss_weights: Tuple[float, float, float] = (1.0, 0.5, 1.0),
    ) -> dict:
        """Return dict of losses and predictions."""

        # Composer ------------------------------------------------------
        ph_logits, bl_logits, role_w, block_mem = self.composer(text_ids, duration_ids)
        block_mem = block_mem[:, : latents.size(1)]  # align length

        # Decoder -------------------------------------------------------
        recon = self.decoder(latents, block_mem)

        # (1) Phoneme CE
        B, S, P, V = ph_logits.shape  # (B,S,phoneme_len,vocab)
        ph_pred = ph_logits.permute(0, 1, 3, 2).contiguous().view(-1, V)  # (B*S*phoneme_len, V)
        ph_true = phoneme_tgt.view(-1)
        loss_ph = F.cross_entropy(ph_pred, ph_true, ignore_index=0)

        # (2) Blank BCE
        loss_bl = F.binary_cross_entropy_with_logits(bl_logits, silence_mask)

        # (3) Latent residual L1
        res_pred = recon[:, 1:] - latents[:, :-1]
        res_true = latents[:, 1:] - latents[:, :-1]
        loss_lat = F.l1_loss(res_pred, res_true)

        w_ph, w_bl, w_lat = loss_weights
        total = w_ph * loss_ph + w_bl * loss_bl + w_lat * loss_lat

        return {
            "total_loss": total,
            "loss_phoneme": loss_ph.detach(),
            "loss_blank": loss_bl.detach(),
            "loss_latent": loss_lat.detach(),
            "phoneme_logits": ph_logits,
            "blank_logits": bl_logits,
            "role_weights": role_w,
            "recon_latents": recon,
        }

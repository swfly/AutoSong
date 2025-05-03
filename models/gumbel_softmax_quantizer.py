import torch
import torch.nn as nn
import torch.nn.functional as F

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"Inf detected in {name}")
    
def safe_gumbel_softmax(logits, tau, hard, dim=-1, eps=1e-6):
    # sample U in (eps , 1-eps)
    U = torch.rand_like(logits).clamp_(min=eps, max=1. - eps)
    g = -torch.log(-torch.log(U))          # finite
    y = torch.softmax((logits + g) / tau, dim=dim)

    if hard:
        # straight-through argmax
        idx  = y.argmax(dim, keepdim=True)
        y_h  = torch.zeros_like(logits).scatter_(dim, idx, 1.)
        y    = (y_h - y).detach() + y
    return y
class GumbelSoftmaxQuantizer(nn.Module):
    """
    Discrete bottleneck with Gumbel-Softmax.
    • Keeps one codebook *per channel*  → shape (C, K, D)
    • Can be initialised from an existing VectorQuantizerEMA weight tensor
      so old checkpoints continue to load.
    
    Args
    ----
    num_embeddings : K     (vocabulary size per codebook)
    patch_hw       : D     (H*W of one flattened patch)
    num_channels   : C
    tau_init       : initial temperature
    tau_min        : lower temperature bound during annealing
    anneal_rate    : multiplicative decay applied every .anneal() call
    hard           : if True, straight-through hard one-hot at forward()
    """
    def __init__(
        self,
        num_embeddings: int,
        patch_hw: int,
        num_channels: int,
        tau_init: float = 1.0,
        tau_min: float = 0.2,
        anneal_rate: float = 0.9996,
        hard: bool = True,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = patch_hw
        self.C = num_channels

        # codebook – *trainable* this time
        embed = torch.randn(self.C, self.K, self.D)
        self.embedding = nn.Parameter(embed)          # <-- now a Parameter

        # temperature schedule
        self.register_buffer("tau", torch.tensor(tau_init))
        self.tau_min      = tau_min
        self.anneal_rate  = anneal_rate
        self.hard         = hard

    # --------------------------------------------------------------
    def forward(self, z: torch.Tensor):
        B, C, H, W = z.shape
        assert C == self.C and H * W == self.D

        z_flat = z.view(B, C, -1)                         # (B,C,D)
        e = self.embedding                                # (C,K,D)

        # negative squared-distance → logits
        x2 = (z_flat ** 2).sum(-1, keepdim=True)          # (B,C,1)
        e2 = (e ** 2).sum(-1).unsqueeze(0)                # (1,C,K)
        xe = torch.einsum("bcd,ckd->bck", z_flat, e)      # (B,C,K)
        logits = -(x2 - 2 * xe + e2).clamp(-5.0, 5.0)     # (B,C,K)
        # ***   more robust Gumbel-Softmax   ***
        gs = safe_gumbel_softmax(
            logits,
            tau=self.tau.item(),
            hard=self.hard,
            dim=-1,
            eps=5e-3,        # <- prevents -inf in log()
        )                                                        # (B,C,K)

        quant = torch.einsum("bck,ckd->bcd", gs, e).view(B,C,H,W)
        # correct straight-through: identity gradient wrt z
        q_st = z + (quant - z).detach()
        indices = gs.argmax(-1)                         # (B,C)
        commit_loss = torch.tensor(0.0, device=z.device)
        self.anneal()
        return q_st, commit_loss, indices

    # --------------------------------------------------------------
    @torch.no_grad()
    def anneal(self):
        """Call once per training step to cool down tau."""
        new_tau = max(self.tau_min, self.tau.item() * self.anneal_rate)
        self.tau.fill_(new_tau)

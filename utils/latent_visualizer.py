import matplotlib.pyplot as plt
import torch

def visualize_latents(latents: list[torch.Tensor], window_names: list[str]):
    """
    Display each latent tensor (B, C, H, W) in its own window.
    Only the first sample in batch is shown.
    Assumes C == 4 for now.
    
    Args:
        latents: list of tensors, each (1, 4, H, W)
        window_names: list of names for figure windows
    """
    assert len(latents) == len(window_names), "Mismatch in number of latents and window names"

    for i, latent in enumerate(latents):
        assert latent.dim() == 4, "Expected (B, C, H, W)"
        b, c, h, w = latent.shape
        assert c == 4, f"Expected 4 channels, got {c}"

        latent = latent[0].detach().cpu().float()  # First sample

        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle(window_names[i])

        for ch in range(4):
            axs[ch].imshow(latent[ch], cmap="viridis", aspect="auto")
            axs[ch].set_title(f"Channel {ch}")
            axs[ch].axis("off")

    plt.show()

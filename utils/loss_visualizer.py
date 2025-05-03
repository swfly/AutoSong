import torch
import matplotlib.pyplot as plt
import argparse

def plot_train_losses(ckpt_path):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Extract the train_losses
    train_losses = checkpoint.get("train_losses", None)

    if train_losses is None:
        print("No 'train_losses' found in the checkpoint.")
        return

    # Plot the training loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_train_losses("checkpoints/train_latent_transformer.pt")

"""
Training Script for Oil Spill Detection Model
Implements training loop with Dice Loss and validation monitoring
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import AttentionUNet
from dataset import OilSpillDataset


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Handles class imbalance well (oil spill pixels << background pixels).

    Formula: 1 - (2 * intersection + smooth) / (sum + smooth)
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]

        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate intersection
        intersection = (pred_flat * target_flat).sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # Return loss (1 - dice)
        return 1 - dice


def train_model(
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    device=None
):
    """
    Main training function.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use (cuda/cpu)
    """

    print("=" * 70)
    print("OIL SPILL DETECTION - TRAINING")
    print("=" * 70)

    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = OilSpillDataset('data/train/images', 'data/train/masks')
    val_dataset = OilSpillDataset('data/val/images', 'data/val/masks')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")

    # Initialize model
    print("\nInitializing Attention U-Net model...")
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss function (Dice Loss)
    criterion = DiceLoss(smooth=1.0)

    # Optimizer (Adam)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    # Training loop
    for epoch in range(num_epochs):
        # ============ Training Phase ============
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False
        )

        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            train_loss += loss.item()
            train_steps += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average training loss
        train_loss /= train_steps
        train_losses.append(train_loss)

        # ============ Validation Phase ============
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            pbar_val = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ",
                leave=False
            )

            for images, masks, _ in pbar_val:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Track loss
                val_loss += loss.item()
                val_steps += 1

                pbar_val.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average validation loss
        val_loss /= val_steps
        val_losses.append(val_loss)

        # Print epoch results
        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            end=""
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(" | ✓ Best model saved", end="")

        print()  # New line

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: best_model.pth")

    # Plot training history
    print("\nGenerating training curves...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss', marker='s')
    plt.axhline(y=best_val_loss, color='r', linestyle='--', label=f'Best Val Loss ({best_val_loss:.4f})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Loss', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training curves saved: training_history.png")

    print("\n✅ All done!")
    print("\nNext steps:")
    print("1. Check training_history.png")
    print("2. Run: python evaluate.py")
    print("3. Run: streamlit run app.py")


if __name__ == '__main__':
    # Train with default parameters (batch_size=4 for 4GB VRAM)
    train_model(
        num_epochs=20,
        batch_size=4,
        learning_rate=1e-4
    )

"""
Evaluation Script for Oil Spill Detection Model
Calculate comprehensive metrics on test set
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import AttentionUNet
from dataset import OilSpillDataset


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate comprehensive segmentation metrics.

    Args:
        pred: Predicted probabilities [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        threshold: Threshold for binarization

    Returns:
        Dictionary with metrics (Dice, IoU, Precision, Recall)
    """
    # Binarize predictions
    pred = (pred > threshold).float()
    target = target.float()

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # True Positives, False Positives, False Negatives
    tp = (pred_flat * target_flat).sum()
    fp = ((1 - target_flat) * pred_flat).sum()
    fn = (target_flat * (1 - pred_flat)).sum()

    # Dice Coefficient
    dice = (2. * tp + 1e-8) / (2. * tp + fp + fn + 1e-8)

    # IoU (Intersection over Union)
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + 1e-8) / (union + 1e-8)

    # Precision
    precision = (tp + 1e-8) / (tp + fp + 1e-8)

    # Recall (Sensitivity)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def evaluate_model(model_path='best_model.pth', device=None):
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to saved model weights
        device: Device to use (cuda/cpu)
    """

    print("=" * 70)
    print("OIL SPILL DETECTION - EVALUATION")
    print("=" * 70)

    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = OilSpillDataset('data/test/images', 'data/test/masks')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for detailed analysis
        shuffle=False
    )
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")

        for images, masks, filenames in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Calculate metrics
            metrics = calculate_metrics(probs, masks)

            all_dice.append(metrics['dice'])
            all_iou.append(metrics['iou'])
            all_precision.append(metrics['precision'])
            all_recall.append(metrics['recall'])

            # Update progress bar
            pbar.set_postfix({
                'Dice': f"{metrics['dice']:.3f}",
                'IoU': f"{metrics['iou']:.3f}"
            })

    # Calculate aggregate statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    avg_dice = np.mean(all_dice)
    avg_iou = np.mean(all_iou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)

    std_dice = np.std(all_dice)
    std_iou = np.std(all_iou)

    print(f"\n📊 Average Metrics:")
    print(f"  Dice Coefficient: {avg_dice:.4f} (±{std_dice:.4f})")
    print(f"  IoU (Jaccard):    {avg_iou:.4f} (±{std_iou:.4f})")
    print(f"  Precision:        {avg_precision:.4f}")
    print(f"  Recall:           {avg_recall:.4f}")

    # F1 Score (same as Dice for binary segmentation)
    f1_score = avg_dice
    print(f"  F1 Score:         {f1_score:.4f}")

    # Calculate additional statistics
    print(f"\n📏 Performance Summary:")
    print(f"  Best Dice:  {max(all_dice):.4f}")
    print(f"  Worst Dice: {min(all_dice):.4f}")
    print(f"  Median IoU: {np.median(all_iou):.4f}")

    # Visualize results distribution
    print("\nGenerating evaluation plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Dice and IoU distribution
    axes[0].hist(all_dice, bins=30, alpha=0.7, label='Dice', color='blue', edgecolor='black')
    axes[0].hist(all_iou, bins=30, alpha=0.7, label='IoU', color='green', edgecolor='black')
    axes[0].axvline(avg_dice, color='blue', linestyle='--', linewidth=2, label=f'Avg Dice ({avg_dice:.3f})')
    axes[0].axvline(avg_iou, color='green', linestyle='--', linewidth=2, label=f'Avg IoU ({avg_iou:.3f})')
    axes[0].set_xlabel('Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Metrics Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    metrics_data = [all_dice, all_iou, all_precision, all_recall]
    axes[1].boxplot(metrics_data, labels=['Dice', 'IoU', 'Precision', 'Recall'])
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    print("✓ Evaluation plots saved: evaluation_results.png")

    # Performance assessment
    print("\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Dice Coefficient: {avg_dice:.4f}")
    print(f"  IoU (Jaccard):    {avg_iou:.4f}")
    print(f"  Precision:        {avg_precision:.4f}")
    print(f"  Recall:           {avg_recall:.4f}")

    if avg_dice >= 0.75:
        print(f"\n✅ EXCELLENT! Outstanding model performance!")
    elif avg_dice >= 0.65:
        print(f"\n✅ VERY GOOD! Strong model performance!")
    elif avg_dice >= 0.60:
        print(f"\n✓ GOOD! Solid model performance!")
    else:
        print(f"\n⚠️  Consider training longer or adjusting hyperparameters.")

    print("\n" + "=" * 70)
    print("✅ Evaluation completed!")
    print("=" * 70)

    print("\nNext step:")
    print("  Run: streamlit run app.py")
    print("=" * 70)


if __name__ == '__main__':
    evaluate_model()

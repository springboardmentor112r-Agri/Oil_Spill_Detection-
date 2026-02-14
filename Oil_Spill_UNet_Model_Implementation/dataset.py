"""
Dataset Class for Oil Spill Detection
Simple preprocessing following proven approach
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for oil spill detection from satellite imagery.

    Preprocessing pipeline:
    1. Grayscale conversion
    2. Resize to 256x256
    3. Median blur (noise reduction)
    4. Normalization to [0, 1]

    Args:
        images_dir: Directory containing input images
        masks_dir: Directory containing segmentation masks
        transform: Optional augmentation transforms
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Get all image files
        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_dir}")

        print(f"Loaded {len(self.images)} images from {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image-mask pair.

        Returns:
            image: Preprocessed image tensor [1, 256, 256]
            mask: Binary mask tensor [1, 256, 256]
            filename: Image filename for reference
        """
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # Read as grayscale (SAR images are single channel)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        # Resize to standard size
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Preprocessing pipeline
        # 1. Median blur to reduce speckle noise (common in SAR images)
        image = cv2.medianBlur(image, 3)

        # 2. Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # 3. Binarize mask (ensure it's 0 or 1)
        mask = (mask > 127).astype(np.float32)

        # Add channel dimension [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Apply transforms if any (for data augmentation during training)
        if self.transform:
            # Note: transforms would need to be applied to both image and mask
            pass

        return image, mask, img_name


def test_dataset():
    """Test dataset loading"""
    print("Testing OilSpillDataset...")

    # Test with train data
    try:
        dataset = OilSpillDataset('data/train/images', 'data/train/masks')

        print(f"\nDataset size: {len(dataset)}")

        # Load first sample
        if len(dataset) > 0:
            image, mask, filename = dataset[0]

            print(f"\nFirst sample:")
            print(f"  Filename: {filename}")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.4f}, {image.max():.4f}]")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask dtype: {mask.dtype}")
            print(f"  Mask unique values: {torch.unique(mask).tolist()}")
            print(f"  Oil spill pixels: {mask.sum().item()}")

            print("\n✅ Dataset test passed!")
        else:
            print("\n⚠️  Dataset is empty. Make sure to run organize_data.py first!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease make sure:")
        print("1. Dataset is downloaded")
        print("2. organize_data.py has been run")
        print("3. data/train/images and data/train/masks exist")


if __name__ == '__main__':
    test_dataset()

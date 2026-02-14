"""
Data Organization Script
Splits dataset into train/val/test (70/15/15)
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def organize_dataset():
    """Split data into train/val/test with stratification"""

    print("=" * 60)
    print("ORGANIZING DATASET")
    print("=" * 60)

    # Paths
    raw_images = "data/raw/images"
    raw_masks = "data/raw/masks"

    # Check if raw data exists
    if not os.path.exists(raw_images):
        print(f"\n❌ Error: {raw_images} not found!")
        print("\nPlease:")
        print("1. Download dataset from Kaggle: https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection")
        print("2. Extract to data/raw/ folder")
        print("3. Make sure you have:")
        print("   - data/raw/images/")
        print("   - data/raw/masks/")
        return

    # Get all images
    all_images = sorted([f for f in os.listdir(raw_images)
                        if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(all_images) == 0:
        print(f"\n❌ Error: No images found in {raw_images}")
        return

    print(f"\nFound {len(all_images)} images")

    # Check if masks exist
    missing_masks = []
    for img in all_images:
        mask_path = os.path.join(raw_masks, img)
        if not os.path.exists(mask_path):
            missing_masks.append(img)

    if missing_masks:
        print(f"\n⚠️  Warning: {len(missing_masks)} images have no corresponding masks")
        # Remove images without masks
        all_images = [img for img in all_images if img not in missing_masks]
        print(f"Using {len(all_images)} images with masks")

    # Split: 70% train, 15% val, 15% test
    print("\nSplitting dataset (70/15/15)...")
    train_imgs, temp_imgs = train_test_split(
        all_images,
        test_size=0.3,
        random_state=42
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=42
    )

    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val:   {len(val_imgs)} images")
    print(f"  Test:  {len(test_imgs)} images")

    # Create folders
    print("\nCreating folders...")
    for split in ['train', 'val', 'test']:
        os.makedirs(f"data/{split}/images", exist_ok=True)
        os.makedirs(f"data/{split}/masks", exist_ok=True)

    # Copy files
    print("\nCopying files...")

    print("  Copying train set...")
    for img in tqdm(train_imgs):
        shutil.copy(
            f"{raw_images}/{img}",
            f"data/train/images/{img}"
        )
        shutil.copy(
            f"{raw_masks}/{img}",
            f"data/train/masks/{img}"
        )

    print("  Copying validation set...")
    for img in tqdm(val_imgs):
        shutil.copy(
            f"{raw_images}/{img}",
            f"data/val/images/{img}"
        )
        shutil.copy(
            f"{raw_masks}/{img}",
            f"data/val/masks/{img}"
        )

    print("  Copying test set...")
    for img in tqdm(test_imgs):
        shutil.copy(
            f"{raw_images}/{img}",
            f"data/test/images/{img}"
        )
        shutil.copy(
            f"{raw_masks}/{img}",
            f"data/test/masks/{img}"
        )

    # Verify
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        img_count = len(os.listdir(f"data/{split}/images"))
        mask_count = len(os.listdir(f"data/{split}/masks"))
        print(f"{split.capitalize()}: {img_count} images, {mask_count} masks")

    print("\n✅ Dataset organization complete!")
    print("\nNext steps:")
    print("1. Run: python train.py")
    print("=" * 60)

if __name__ == '__main__':
    organize_dataset()

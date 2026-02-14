"""
Reorganize Existing Oil Spill Dataset
Takes the existing dataset with train/val and creates proper train/val/test split (70/15/15)
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Source dataset location
SOURCE_DATA_DIR = r"C:\Users\nishu\Documents\springboard\AI_OilSpill_Detection\data"

# Target location (our project)
TARGET_DATA_DIR = "data"

def reorganize_dataset():
    """
    Reorganize dataset from existing train/val structure to train/val/test (70/15/15).
    """

    print("=" * 70)
    print("REORGANIZING OIL SPILL DATASET")
    print("=" * 70)

    # Check if source exists
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"\nError: Source directory not found: {SOURCE_DATA_DIR}")
        return

    print(f"\nSource: {SOURCE_DATA_DIR}")
    print(f"Target: {TARGET_DATA_DIR}")

    # Collect all image filenames from both train and val
    print("\nCollecting all images...")

    train_images_dir = os.path.join(SOURCE_DATA_DIR, "train", "images")
    val_images_dir = os.path.join(SOURCE_DATA_DIR, "val", "images")

    all_images = []

    # Collect from train
    if os.path.exists(train_images_dir):
        train_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]
        all_images.extend([('train', f) for f in train_files])
        print(f"  Train images: {len(train_files)}")

    # Collect from val
    if os.path.exists(val_images_dir):
        val_files = [f for f in os.listdir(val_images_dir) if f.endswith('.png')]
        all_images.extend([('val', f) for f in val_files])
        print(f"  Val images: {len(val_files)}")

    total_images = len(all_images)
    print(f"\n  Total images: {total_images}")

    if total_images == 0:
        print("\nNo images found in source directory!")
        return

    # Split into train/val/test (70/15/15)
    print("\nSplitting dataset (70/15/15)...")

    # First split: 70% train, 30% temp
    train_split, temp_split = train_test_split(
        all_images,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    # Second split: 15% val, 15% test (50/50 from temp)
    val_split, test_split = train_test_split(
        temp_split,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    print(f"  Train: {len(train_split)} images ({len(train_split)/total_images*100:.1f}%)")
    print(f"  Val:   {len(val_split)} images ({len(val_split)/total_images*100:.1f}%)")
    print(f"  Test:  {len(test_split)} images ({len(test_split)/total_images*100:.1f}%)")

    # Create target directory structure
    print("\nCreating directory structure...")
    for split_name in ['train', 'val', 'test']:
        for folder in ['images', 'masks']:
            path = os.path.join(TARGET_DATA_DIR, split_name, folder)
            os.makedirs(path, exist_ok=True)

    print("  Directories created successfully")

    # Copy files to new structure
    print("\nCopying files...")

    splits = {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }

    # Track used filenames to prevent collisions
    used_filenames = {}

    for split_name, file_list in splits.items():
        print(f"\n  Copying {split_name} set...")
        used_filenames[split_name] = set()

        for source_split, filename in tqdm(file_list, desc=f"  {split_name}"):
            # Source paths
            src_img = os.path.join(SOURCE_DATA_DIR, source_split, "images", filename)
            src_mask = os.path.join(SOURCE_DATA_DIR, source_split, "masks", filename)

            # Handle filename collisions by prefixing with source folder
            target_filename = filename
            if target_filename in used_filenames[split_name]:
                # Add source prefix to prevent collision
                base, ext = os.path.splitext(filename)
                target_filename = f"{source_split}_{base}{ext}"

            used_filenames[split_name].add(target_filename)

            # Target paths
            dst_img = os.path.join(TARGET_DATA_DIR, split_name, "images", target_filename)
            dst_mask = os.path.join(TARGET_DATA_DIR, split_name, "masks", target_filename)

            # Copy image
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"\n    Warning: Image not found: {src_img}")

            # Copy mask
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"\n    Warning: Mask not found: {src_mask}")

    # Verify the copy
    print("\n\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    for split_name in ['train', 'val', 'test']:
        img_dir = os.path.join(TARGET_DATA_DIR, split_name, "images")
        mask_dir = os.path.join(TARGET_DATA_DIR, split_name, "masks")

        num_images = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
        num_masks = len([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        print(f"\n{split_name.capitalize():5s}: {num_images} images, {num_masks} masks", end="")

        if num_images == num_masks:
            print(" [OK]")
        else:
            print(" [MISMATCH!]")

    print("\n" + "=" * 70)
    print("DATASET REORGANIZATION COMPLETE!")
    print("=" * 70)

    print("\nDataset structure:")
    print(f"  {TARGET_DATA_DIR}/")
    print(f"    ├── train/ ({len(train_split)} samples)")
    print(f"    │   ├── images/")
    print(f"    │   └── masks/")
    print(f"    ├── val/ ({len(val_split)} samples)")
    print(f"    │   ├── images/")
    print(f"    │   └── masks/")
    print(f"    └── test/ ({len(test_split)} samples)")
    print(f"        ├── images/")
    print(f"        └── masks/")

    print("\nNext steps:")
    print("  1. Verify dataset: python dataset.py")
    print("  2. Start training: python train.py")
    print("=" * 70)


if __name__ == '__main__':
    reorganize_dataset()

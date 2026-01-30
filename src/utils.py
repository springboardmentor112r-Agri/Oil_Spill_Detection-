import cv2
import numpy as np
import os


IMG_SIZE = (128, 128)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    return image


def preprocess_for_model(image_path):
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)
    return image


def postprocess_mask(mask, threshold=0.5):
    mask = (mask > threshold).astype("uint8")
    mask = mask.squeeze()
    return mask


def save_mask(mask, save_path):
    mask = (mask * 255).astype("uint8")
    cv2.imwrite(save_path, mask)


def overlay_mask(image, mask, alpha=0.4):
    overlay = image.copy()
    overlay[mask == 1] = [255, 0, 0]
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

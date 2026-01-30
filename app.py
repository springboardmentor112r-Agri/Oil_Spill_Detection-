import os
import cv2
import numpy as np
import tensorflow as tf
from src.utils import preprocess_for_model, postprocess_mask, save_mask, overlay_mask
from src.utils import load_image


MODEL_PATH = "results/model/oil_spill_unet"
INPUT_IMAGE_PATH = "sample_input.png"
OUTPUT_DIR = "results/app_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"dice_coefficient": dice_coefficient}
    )

    print("Preprocessing input image...")
    input_tensor = preprocess_for_model(INPUT_IMAGE_PATH)

    print("Running prediction...")
    pred = model.predict(input_tensor)[0]

    mask = postprocess_mask(pred)

    original_image = load_image(INPUT_IMAGE_PATH)
    original_image = (original_image * 255).astype("uint8")

    mask_path = os.path.join(OUTPUT_DIR, "predicted_mask.png")
    overlay_path = os.path.join(OUTPUT_DIR, "overlay.png")

    save_mask(mask, mask_path)

    overlay = overlay_mask(original_image, mask)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Inference complete.")
    print(f"Saved mask to: {mask_path}")
    print(f"Saved overlay to: {overlay_path}")


if __name__ == "__main__":
    main()

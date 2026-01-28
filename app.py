import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import os
from io import BytesIO

# --- Configuration --- #
TARGET_SIZE = (128, 128)

# ✅ Exact filename from repo
MODEL_FILENAME = "unet_oil_spill_segmentation_model_128x128 (3).keras"

DEFAULT_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), MODEL_FILENAME),
    os.path.join(os.path.dirname(__file__), "models", MODEL_FILENAME),
]

# --- Helper Functions --- #
@st.cache_resource
def load_model_from_path(model_path: str):
    if not model_path:
        return None
    try:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.exception(f"Error loading model from {model_path}: {e}")
        return None


def find_existing_model_path():
    for path in DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def preprocess_image_for_prediction(image):
    # ✅ Force NumPy + float32 (FIX)
    img = np.array(image.convert("L"), dtype=np.float32)
    img = PIL.Image.fromarray(img.astype(np.uint8)).resize(
        TARGET_SIZE, PIL.Image.LANCZOS
    )
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


def postprocess_mask_prediction(prediction, original_size):
    mask = (prediction.squeeze() > 0.5).astype(np.uint8) * 255
    mask = PIL.Image.fromarray(mask).resize(
        original_size, PIL.Image.NEAREST
    )
    return np.array(mask)


def overlay_mask_on_image(image_rgb, mask, color=(0, 255, 0), alpha=0.5):
    overlay = np.zeros_like(image_rgb)
    mask_bool = mask > 0

    for c in range(3):
        overlay[..., c][mask_bool] = color[c]

    blended = image_rgb * (1 - alpha) + overlay * alpha
    return blended.astype(np.uint8)


# --- Streamlit UI --- #
def app_main():
    st.title("Oil Spill Detection Application")
    st.write("Upload a SAR image to detect oil spills.")

    model = None

    # Load model from repo
    model_path = find_existing_model_path()
    if model_path:
        st.success(f"Loaded model: {os.path.basename(model_path)}")
        model = load_model_from_path(model_path)
    else:
        st.error("Model file not found in repository.")
        return

    uploaded_file = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting oil spill..."):
            processed = preprocess_image_for_prediction(image)
            prediction = model.predict(processed)
            mask = postprocess_mask_prediction(prediction, image.size)

            if np.any(mask > 0):
                st.success("Oil Spill Detected!")
            else:
                st.info("No Oil Spill Detected.")

            overlay = overlay_mask_on_image(
                np.array(image.convert("RGB")),
                mask
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(mask, caption="Predicted Mask", use_column_width=True)
            with col2:
                st.image(overlay, caption="Overlay Result", use_column_width=True)

            # Download
            st.subheader("Download Results")
            buf = BytesIO()
            PIL.Image.fromarray(mask).save(buf, format="PNG")
            st.download_button(
                "Download Mask",
                buf.getvalue(),
                "predicted_mask.png",
                "image/png"
            )


if __name__ == "__main__":
    app_main()

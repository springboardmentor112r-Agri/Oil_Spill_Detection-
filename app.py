import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# =========================================================
# Streamlit Page Config
# =========================================================
st.set_page_config(
    page_title="Oil Spill Detection",
    layout="centered"
)

st.title("🛢️ Oil Spill Detection using Satellite Images")
st.write("Upload a satellite image to detect oil spill regions in real-time.")

# =========================================================
# Custom Loss & Metrics (USED DURING TRAINING)
# =========================================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) +
        tf.keras.backend.sum(y_pred_f) +
        smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

# =========================================================
# Load Trained Model (SAFE + CACHED)
# =========================================================
@st.cache_resource
def load_trained_model():
    model_path = Path("models/unet_oil_spill_model.h5")

    if not model_path.exists():
        st.error("❌ Model file not found in 'models/' folder")
        st.stop()

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                "dice_loss": dice_loss,
                "dice_coefficient": dice_coefficient
            }
        )
    except Exception as e:
        st.error("❌ Failed to load model. Check logs.")
        st.exception(e)
        st.stop()

    return model
model = load_trained_model()


# =========================================================
# Image Upload
# =========================================================
uploaded_file = st.file_uploader(
    "Upload Satellite Image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

# =========================================================
# Inference Pipeline
# =========================================================
if uploaded_file is not None:
    try:
        # Read image safely using PIL (NO OpenCV)
        image = Image.open(uploaded_file).convert("L")
        image = np.array(image)
    except Exception:
        st.error("❌ Unable to read the uploaded image")
        st.stop()

    st.subheader("Original Image")
    st.image(image, clamp=True)

    # -------------------------
    # Preprocessing
    # -------------------------
    IMG_SIZE = 256

    image_resized = Image.fromarray(image).resize(
        (IMG_SIZE, IMG_SIZE),
        resample=Image.BILINEAR
    )

    image_resized = np.array(image_resized, dtype=np.float32) / 255.0

    # Model input shape: (1, 256, 256, 1)
    image_input = image_resized[np.newaxis, ..., np.newaxis]

    # -------------------------
    # Prediction
    # -------------------------
    with st.spinner("🔍 Detecting oil spill regions..."):
        prediction = model.predict(image_input, verbose=0)

    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)

    # -------------------------
    # Results
    # -------------------------
    st.subheader("Predicted Oil Spill Mask")
    st.image(mask * 255, clamp=True)

    st.subheader("Overlay Visualization")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_resized, cmap="gray")
    ax.imshow(mask, cmap="jet", alpha=0.5)
    ax.axis("off")
    st.pyplot(fig)

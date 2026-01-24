import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import rasterio
import imageio.v2 as imageio
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🛢️ AI-Based Oil Spill Detection (SAR Images)")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_unet():
    return load_model("oil_spill_unet.keras", compile=False)

model = load_unet()

# -----------------------------
# Preprocess Image
# -----------------------------
def preprocess_uploaded_image(uploaded_file, size=(256, 256)):
    filename = uploaded_file.name.lower()
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        with rasterio.open(uploaded_file) as src:
            img = src.read(1)
    else:
        img = imageio.imread(uploaded_file)
        if img.ndim == 3:
            img = img[:, :, 0]
    img = cv2.resize(img, size)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# -----------------------------
# Land Mask
# -----------------------------
def generate_land_mask(img):
    land = img > np.percentile(img, 75)
    land = (land.astype(np.uint8)) * 255
    kernel = np.ones((5, 5), np.uint8)
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, kernel)
    return land

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload SAR Image (.tif, .png, .jpg)", 
                                 type=["tif","tiff","png","jpg","jpeg"])

if uploaded_file:

    # Preprocess
    img = preprocess_uploaded_image(uploaded_file)
    img_input = img[np.newaxis, ..., np.newaxis]

    # Predict
    pred = model.predict(img_input)[0, :, :, 0]
    pred_bin = (pred > 0.3).astype(np.uint8) * 255

    # Land mask
    land_mask = generate_land_mask(img)
    final_mask = cv2.bitwise_and(pred_bin, cv2.bitwise_not(land_mask))

    # -------------------
    # Remove tiny speckles
    # -------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    final_oil_mask = np.zeros_like(final_mask)
    min_region_size = 500  # adjust based on SAR resolution
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            final_oil_mask[labels == i] = 255

    # -------------------
    # Oil Percentage
    # -------------------
    oil_pixels = np.sum(final_oil_mask == 255)
    total_pixels = final_oil_mask.size
    oil_percentage = (oil_pixels / total_pixels) * 100

    # -------------------
    # Final Decision
    # -------------------
    oil_detected = oil_percentage >= 2.0  # noise filtered
    if not oil_detected:
        final_oil_mask[:] = 0
        oil_percentage = 0.0

    # -------------------
    # Save Results
    # -------------------
    SAVE_DIR = "results"
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save original uploaded image
    with open(os.path.join(SAVE_DIR, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Save predicted mask
    mask_filename = uploaded_file.name.replace(".tif","_mask.png").replace(".tiff","_mask.png")
    cv2.imwrite(os.path.join(SAVE_DIR, mask_filename), final_oil_mask)

    # -------------------
    # Display Results
    # -------------------
    st.subheader("🔍 Prediction Result")
    if oil_detected:
        st.error("🚨 OIL SPILL DETECTED")
    else:
        st.success("✅ NO OIL SPILL DETECTED")

    st.metric("🛢️ Oil Spill Area (%)", f"{oil_percentage:.2f}%")

    # -------------------
    # Visualization
    # -------------------
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].set_title("Preprocessed SAR Image"); ax[0].imshow(img,cmap="gray"); ax[0].axis("off")
    ax[1].set_title("Land Mask (White = Land)"); ax[1].imshow(land_mask,cmap="gray"); ax[1].axis("off")
    ax[2].set_title("Final Oil Mask (White = Oil)"); ax[2].imshow(final_oil_mask,cmap="gray",vmin=0,vmax=255); ax[2].axis("off")
    st.pyplot(fig)

    # Optional Alert
    if oil_detected:
        st.warning("🚨 ALERT: Oil spill detected! Saved results to 'results/' folder.")

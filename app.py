import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("🛢️ Oil Spill Detection System")

# Load trained Keras classification model
model = load_model("my_classification_model.keras")

file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if file:
    # Read image bytes
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not load image. Please upload a valid image file.")
    else:
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # =========================
        # 🔹 PREPROCESSING PIPELINE
        # =========================

        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. STRONG DENOISING (Non-Local Means)
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=15,          # Filter strength (10–20 recommended)
            templateWindowSize=7,
            searchWindowSize=21
        )

        # 3. Resize to model input size
        resized = cv2.resize(denoised, (256, 256))

        # 4. Normalize pixel values
        normalized = resized / 255.0

        # 5. Add channel + batch dimensions
        processed_img = np.expand_dims(normalized, axis=-1)
        processed_img = np.expand_dims(processed_img, axis=0)

        # =========================
        # 🔹 MODEL PREDICTION
        # =========================

        prediction = model.predict(processed_img)[0][0]

        predicted_class = "Oil Spill" if prediction >= 0.5 else "Non Oil Spill"
        confidence = prediction if predicted_class == "Oil Spill" else (1 - prediction)

        # =========================
        # 🔹 OUTPUT
        # =========================

        st.subheader("🔍 Prediction Result")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence ({predicted_class}):** {confidence:.4f}")

        # Optional: Show denoised image
        st.image(denoised, caption="Denoised Image (Used for Prediction)", use_column_width=True)

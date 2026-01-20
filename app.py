import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("Oil Spill Detection System")

# Load the trained Keras classification model
# Using 'my_classification_model.keras' as it's the recommended new format
model = load_model("my_classification_model.keras")

file = st.file_uploader("Upload Satellite Image")

if file:
    # Read the image as a byte stream and then convert to numpy array
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8);
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Load as color image for display

    if img is None:
        st.error("Could not load image. Please ensure it's a valid image file.")
    else:
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image for model prediction
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # b. Convert to grayscale
        processed_img = cv2.resize(processed_img, (256, 256)) # c. Resize to (256, 256)
        processed_img = processed_img / 255.0 # d. Normalize pixel values
        processed_img = np.expand_dims(processed_img, axis=-1) # e. Add channel dimension (256, 256, 1)

        # 4. Expand dimensions for batch prediction (1, 256, 256, 1)
        processed_img_batch = np.expand_dims(processed_img, axis=0)

        # 5. Make prediction to get 'Oil Spill' probability
        prediction = model.predict(processed_img_batch)[0][0] # Get the single scalar prediction

        # 6. Determine the predicted class label
        predicted_class = "Oil Spill" if prediction >= 0.5 else "Non Oil Spill"
        confidence = prediction if predicted_class == "Oil Spill" else (1 - prediction)

        # 7. Display the predicted class and confidence
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence ({predicted_class}):** {confidence:.4f}")

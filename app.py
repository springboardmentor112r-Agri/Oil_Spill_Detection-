import streamlit as st
import torch
import cv2
import numpy as np
from model import OilSpillCNN


# Load model
model = OilSpillCNN()
model.load_state_dict(torch.load("oil_spill_model.pth", map_location=torch.device('cpu')))
model.eval()


=======
>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)


uploaded_file = st.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])


=======
uploaded_file = st.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])

>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


=======

>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)

=======
>>>>>>> 544f254 (Add deployment files & PyTorch model via LFS)
    if output.item() > 0.5:
        st.success("Prediction: Oil Spill Detected")
    else:
        st.success("Prediction: No Oil Spill Detected")

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
import os
from src.utils import preprocess_for_model, postprocess_mask

app = FastAPI(title="Oil Spill Detection API")

MODEL_PATH = "results/model/oil_spill_unet.h5"
UPLOAD_DIR = "results/api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"dice_coefficient": dice_coefficient}
)

from fastapi.middleware.cors import CORSMiddleware
import base64

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image_norm = image / 255.0
    image_input = np.expand_dims(image_norm, axis=0)

    pred = model.predict(image_input)[0]
    mask = (pred > 0.5).astype("uint8")

    overlay = image.copy()
    overlay[mask.squeeze() == 1] = [255, 0, 0]

    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "message": "Prediction successful",
        "image": encoded_image
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

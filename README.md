---

# 🌊 AI-Driven Oil Spill Detection and Monitoring

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![SAR](https://img.shields.io/badge/SAR-Remote%20Sensing-green)

---

## 📌 Project Overview

Oil spills pose severe threats to marine ecosystems, coastal environments, and economies. Traditional monitoring methods rely on manual inspection of satellite imagery, which is time-consuming, labor-intensive, and often delayed.

This project presents an **AI-driven oil spill detection and segmentation system** using **Synthetic Aperture Radar (SAR)** satellite imagery and a **U-Net deep learning architecture**. The system automatically identifies and localizes oil spill regions and is deployed via a **Streamlit web application** for real-time inference and visualization.

🔗 **Live Application**
👉 [https://ai-driven-oil-spill-detection-and-monitorizing-phaa.streamlit.app/](https://ai-driven-oil-spill-detection-and-monitorizing-phaa.streamlit.app/)

---

## 🧠 System Architecture

The system follows a modular, end-to-end pipeline:

1. SAR Image Acquisition
2. SAR-Specific Denoising (Speckle Noise Reduction)
3. Deep Learning-Based Segmentation (U-Net)
4. Mask Generation and Overlay Visualization
5. Web Deployment via Streamlit

This architecture enables automated oil spill monitoring from raw satellite imagery to user-interactive prediction outputs.

---

## 📂 Dataset

* **Dataset Name**: Deep SAR (SOS) Oil Spill Dataset
* **Source**: Kaggle
* **Archived Version**: [https://zenodo.org/records/8346860](https://zenodo.org/records/8346860)
* **Data Type**: SAR images with binary segmentation masks
* **Sensors**: Sentinel-1 and PALSAR

### 📊 Dataset Statistics

#### Initial Dataset (Before Cleanup)

| Split      | Images | Masks |
| ---------- | ------ | ----- |
| Training   | 6455   | 6459  |
| Validation | 1389   | 1615  |

#### Data Cleaning

* ❌ Removed **4** misaligned training masks
* ❌ Removed **226** misaligned validation masks
* ✅ Ensured **1:1 image–mask correspondence**

#### Final Dataset (After Splitting)

| Split      | Image–Mask Pairs |
| ---------- | ---------------- |
| Training   | 5164             |
| Validation | 1389             |
| Test       | 1291             |

🔁 **Augmented Training Set Size**: 10,328 image–mask pairs

---

## 🧪 Data Exploration & Preprocessing

### ✔ Exploratory Analysis

* Visualized representative image–mask pairs
* Studied pixel-level statistical properties of spill and non-spill regions
* Observed **lower radar backscatter intensity in oil spill regions**, a known SAR phenomenon

---

### 📊 Pixel Intensity Analysis: Spill vs. Non-Spill Regions

| Region Type           | Total Pixels | Mean Intensity | Std. Deviation |
| --------------------- | ------------ | -------------- | -------------- |
| **Spill Regions**     | 57,388       | **90.28**      | 51.32          |
| **Non-Spill Regions** | 270,292      | **140.12**     | 43.19          |

**Key Insight:** Oil slicks dampen capillary waves, reducing SAR backscatter and creating strong segmentation cues.

---

### ✔ Preprocessing Steps

* **Resizing**: `128 × 128`
* **Normalization**: Pixel values scaled to `[0, 1]`
* **SAR-Specific Denoising**:

```python
skimage.restoration.denoise_wavelet(
    wavelet="db1",
    mode="soft",
    sigma=0.05
)
```

---

### ✔ Data Augmentation (Training Only)

Implemented using **Albumentations**:

* Horizontal & Vertical Flips
* Rotation (±30°)
* Random Brightness & Contrast

---

## 🧩 Model Architecture (U-Net)

* **Input Shape**: `(128, 128, 1)`
* **Encoder Filters**: `32 → 64 → 128`
* **Bottleneck**: `256`
* **Decoder Filters**: `128 → 64 → 32`
* **Output**: `1×1 Conv + Sigmoid`

Optimized for **accuracy + efficiency** on SAR data.

---

## ⚙️ Training Configuration

* **Framework**: TensorFlow / Keras
* **Optimizer**: Adam (`1e-4`)
* **Loss**: Binary Cross-Entropy
* **Metrics**: Mean IoU, Accuracy, Precision, Recall
* **Batch Size**: 32
* **Epochs**: 50 (Early Stopping)
* **Mixed Precision**: `mixed_float16`

---

## 📈 Model Evaluation (Test Set)

| Metric          | Value  |
| --------------- | ------ |
| Loss            | 0.1992 |
| Mean IoU        | 0.3779 |
| Binary Accuracy | 0.9168 |
| Precision       | 0.8490 |
| Recall          | 0.8035 |

✔ Strong generalization
✔ Balanced precision–recall

---

## 🌐 Deployment (Streamlit App)

### Features

* Upload SAR images
* Real-time inference
* Oil spill detection & segmentation
* Mask overlay visualization
* Download predicted masks

---

## ✅ Requirements & Dependencies

### 🔧 Hardware Requirements

* **GPU access recommended**
* Designed for **Google Colab (CUDA-enabled GPU)**

---

### 🐍 Software Requirements

#### Core Python Libraries

* `os`, `shutil`, `random`, `math`, `io (BytesIO)`

#### Numerical Computing

* `numpy`

#### Image Processing

* `Pillow (PIL)`
* `opencv-python (cv2)`
* `scikit-image` (for `denoise_wavelet`)

#### Data Augmentation

* `albumentations==1.3.1`

#### Deep Learning

* `tensorflow` (with `tensorflow.keras`)

#### Visualization

* `matplotlib`
* `IPython.display`

#### Web Application

* `streamlit`
* `pyngrok`

---

### 📦 Installation (Colab / Local)

```bash
pip install tensorflow numpy pillow albumentations scikit-image \
streamlit pyngrok opencv-python matplotlib
```

---

### 📁 Dataset Requirements

Dataset must be placed in **Google Drive** at:

```text
/content/drive/MyDrive/Deep SAR (SOS) Dataset/
```

Expected structure:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── masks/
│   └── masks/
│       ├── train/
│       └── val/
```

---

### 🔑 Ngrok Authentication

Required for public deployment from Colab:

```bash
ngrok authtoken YOUR_NGROK_AUTH_TOKEN
```

---

## 🚀 Run the App (Colab + LocalTunnel)

```bash
pip install streamlit
npm install -g localtunnel
streamlit run app.py --server.port 8501 & npx localtunnel --port 8501
```

---

## 📁 Project Structure

```text
├── app.py
├── models/
│   └── unet_oil_spill_segmentation_model_128x128.keras
├── dataset_processed_128x128/
│   ├── images/
│   └── masks/
├── notebooks/
│   └── Presentation_and_Documentation.ipynb
├── README.md
```

---

## 🔮 Future Enhancements

* Dice / Focal Loss
* CRF-based post-processing
* Multi-sensor fusion
* Real-time alert APIs
* Confidence estimation

---

## 📜 License

Released under the **MIT License**.

---
Just tell me 👌

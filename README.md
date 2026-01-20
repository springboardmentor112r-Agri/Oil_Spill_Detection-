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
👉 https://ai-driven-oil-spill-detection-and-monitorizing-phaa.streamlit.app/

---

## 🧠 System Architecture

The system follows a modular, end-to-end pipeline:

1. **SAR Image Acquisition**
2. **SAR-Specific Denoising (Speckle Noise Reduction)**
3. **Deep Learning-Based Segmentation (U-Net)**
4. **Mask Generation and Overlay Visualization**
5. **Web Deployment via Streamlit**

This architecture enables automated oil spill monitoring from raw satellite imagery to user-interactive prediction outputs.

---

## 📂 Dataset

- **Dataset Name**: Deep SAR (SOS) Oil Spill Dataset  
- **Source**: Kaggle  
- **Archived Version**: https://zenodo.org/records/8346860  
- **Data Type**: SAR images with binary segmentation masks  
- **Sensors**: Sentinel-1 and PALSAR  

### 📊 Dataset Statistics

#### Initial Dataset (Before Cleanup)

| Split | Images | Masks |
|------|--------|-------|
| Training | 6455 | 6459 |
| Validation | 1389 | 1615 |

#### Data Cleaning

- ❌ Removed **4** misaligned training masks  
- ❌ Removed **226** misaligned validation masks  
- ✅ Ensured **1:1 image–mask correspondence**

#### Final Dataset (After Splitting)

| Split | Image–Mask Pairs |
|------|------------------|
| Training | 5164 |
| Validation | 1389 |
| Test | 1291 |

🔁 **Augmented Training Set Size**: 10,328 image–mask pairs

---

## 🧪 Data Exploration & Preprocessing

### ✔ Exploratory Analysis

- Visualized representative image–mask pairs  
- Studied pixel-level statistical properties of spill and non-spill regions  
- Observed **lower radar backscatter intensity in oil spill regions**, a known SAR phenomenon  

---

### 📊 Pixel Intensity Analysis: Spill vs. Non-Spill Regions

To quantitatively validate SAR characteristics, a detailed **pixel intensity distribution analysis** was conducted using ground-truth masks.

#### Methodology

- Pixels were separated into:
  - **Spill regions** (mask > 0)
  - **Non-spill regions** (mask = 0)
- Aggregated statistics were computed across **multiple representative image–mask pairs**
- Histogram-based comparison was used to analyze separability

#### Aggregated Statistical Results (5 Sample Pairs)

| Region Type | Total Pixels | Mean Intensity | Std. Deviation |
|------------|--------------|----------------|----------------|
| **Spill Regions** | 57,388 | **90.28** | 51.32 |
| **Non-Spill Regions** | 270,292 | **140.12** | 43.19 |

#### Key Observations

- Spill regions show **significantly lower mean intensity**
- Higher variance reflects **irregular spill boundaries and heterogeneous textures**
- Distributions remain **clearly separable despite partial overlap**

These findings confirm that **oil slicks dampen capillary waves**, reducing SAR backscatter and providing strong learnable cues for segmentation.

---

### ✔ Preprocessing Steps

- **Resizing**: `128 × 128`
  - Images: `PIL.Image.LANCZOS`
  - Masks: `PIL.Image.NEAREST`
- **Normalization**: Pixel values scaled to `[0, 1]`
- **SAR-Specific Denoising**:

```python
skimage.restoration.denoise_wavelet(
    wavelet="db1",
    mode="soft",
    sigma=0.05
)
````

### ✔ Data Augmentation (Training Only)

Implemented using **Albumentations**:

* Horizontal and Vertical Flips
* Rotation (±30°)
* Random Brightness and Contrast Adjustments

---

## 🧩 Model Architecture (U-Net)

An **optimized lightweight U-Net** designed specifically for SAR image segmentation.

### Architecture Details

* **Input Shape**: `(128, 128, 1)`
* **Encoder Filters**: `32 → 64 → 128`
* **Bottleneck**: `256`
* **Decoder Filters**: `128 → 64 → 32`
* **Output Layer**: `1×1 Conv + Sigmoid`

This design balances **segmentation accuracy** with **computational efficiency**.

---

## ⚙️ Training Configuration

* **Framework**: TensorFlow / Keras
* **Optimizer**: Adam (`learning_rate = 1e-4`)
* **Loss Function**: Binary Cross-Entropy
* **Metrics**:

  * Mean IoU
  * Binary Accuracy
  * Precision
  * Recall
* **Batch Size**: 32
* **Epochs**: 50 (Early stopping applied)
* **Mixed Precision Training**: `mixed_float16`

### Callbacks

```python
EarlyStopping(patience=10, restore_best_weights=True)
ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
```

---

## 📈 Model Evaluation (Test Set)

| Metric              | Value  |
| ------------------- | ------ |
| **Loss**            | 0.1992 |
| **Mean IoU**        | 0.3779 |
| **Binary Accuracy** | 0.9168 |
| **Precision**       | 0.8490 |
| **Recall**          | 0.8035 |

✔ High pixel-level accuracy
✔ Balanced precision–recall
✔ Good generalization to unseen SAR images

---

## 🖼️ Visualization of Results

* Side-by-side visualization of:

  * Original SAR Image
  * Ground Truth Mask
  * Predicted Mask
* Overlay visualization highlighting oil spill regions in **green**
* Visual outputs saved for reporting and presentation

---

## 🌐 Deployment (Streamlit App)

### Features

* Upload SAR images
* Real-time inference
* Binary classification:

  * **Oil Spill Detected**
  * **No Oil Spill Detected**
* Segmentation mask and overlay visualization
* Download predicted mask

### Classification Logic

```python
is_spill_detected = np.any(predicted_mask > 0)
```

---

## 🚀 Run Locally (Google Colab + LocalTunnel)

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

## 🧪 Results Summary

* High segmentation accuracy (~92% pixel-level)
* Reliable oil spill localization
* Robust against SAR speckle noise
* Fully deployable and reproducible pipeline

---

## 🔮 Future Enhancements

* Dice / Focal Loss for improved IoU
* Morphological and CRF post-processing
* Multi-sensor fusion (SAR + optical)
* Real-time alerting APIs
* Uncertainty and confidence estimation

---

## 🙏 Acknowledgments

* Dataset: Kaggle Deep SAR (SOS)
* Archive: Zenodo
* Libraries: TensorFlow, Albumentations, Streamlit

---

## 📜 License

This project is released under the **MIT License**.

```
```

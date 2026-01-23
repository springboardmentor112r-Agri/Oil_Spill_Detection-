# 🛢️ Oil Spill Detection Using Deep Learning on SAR Imagery

This repository contains an end-to-end implementation of an **Oil Spill Detection system**
using **Synthetic Aperture Radar (SAR) satellite images** and **deep learning-based semantic
segmentation** techniques.

The project is developed as part of an internship and follows a **modular, milestone-based
approach**, covering data exploration, preprocessing, model development, training, evaluation,
and visualization.

---

## 📌 Project Objectives

The primary goals of this project are:

- To analyze satellite SAR imagery for detecting oil spill regions
- To preprocess SAR data and reduce noise artifacts
- To design and train a deep learning segmentation model
- To evaluate model performance using standard segmentation metrics
- To visualize predicted oil spill regions effectively
- To prepare the system for future real-time deployment

---

## 🛰️ Dataset Description

| Attribute | Details |
|---------|--------|
Dataset Name | Sentinel-1 SAR Oil Spill Dataset |
Source | Zenodo |
Image Type | Synthetic Aperture Radar (SAR) |
Image Channels | Single-channel (Grayscale) |
Classes | Oil Spill, Non-Spill, Look-Alike |
Annotations | Pixel-level segmentation masks |

The dataset includes real-world SAR images containing:
- Confirmed oil spill regions
- Clean sea surface (non-spill)
- Look-alike patterns that resemble oil spills

---

## 📂 Repository Structure

Oil_Spill_Detection/
│
├── notebooks/
│ └── Oil_Spill_Detection_End_to_End.ipynb
│
├── models/
│ └── unet_oil_spill_model.h5
│
├── screenshots
│   ├── sample_input_image.png
│   ├── predicted_mask.png
│   └── overlay_visualization.png
├── app.py
│ └── (Deployment files – Module 6 to be added)
│
├── README.md
├── requirements.txt
└── .gitignore


---

## 📒 Notebook Overview

All implemented modules are organized inside a **single, well-structured notebook**:

📄 **`notebooks/Oil_Spill_Detection_End_to_End.ipynb`**

Each module is clearly separated using Markdown headings for easy understanding and review.

---

## 🧩 Module-wise Implementation

### 🔹 Module 2: Data Exploration and Preprocessing

This module focuses on understanding and preparing SAR data for model training.

**Key steps include:**

- Visualization of sample SAR images and masks
- Statistical analysis of pixel intensity distributions
- Image resizing to a standard resolution (256 × 256)
- Pixel normalization for stable model training
- SAR-specific speckle noise reduction
- Data augmentation techniques:
  - Horizontal and vertical flipping
  - Rotation
  - Brightness and contrast variation

---

### 🔹 Module 3: Model Development (Segmentation)

This module implements the deep learning architecture.

| Component | Description |
|---------|------------|
Model Type | U-Net |
Task | Semantic Segmentation |
Input | Single-channel SAR image |
Output | Binary oil spill mask |
Framework | TensorFlow / Keras |

The U-Net architecture follows an encoder–decoder structure optimized for
pixel-level prediction tasks.

---

### 🔹 Module 4: Training and Evaluation

This module handles model training and performance evaluation.

**Loss Functions Used**
- Dice Loss
- Binary Cross-Entropy (BCE)

**Evaluation Metrics**
- Accuracy
- Intersection over Union (IoU)
- Dice Coefficient
- Precision
- Recall

Hyperparameters were fine-tuned based on validation performance to improve
generalization.

---

### 🔹 Module 5: Visualization of Results

This module focuses on interpretability and result presentation.

**Visualizations include:**
- Original SAR image
- Ground truth segmentation mask
- Predicted oil spill mask
- Overlay of prediction on SAR image using color maps

These visual outputs are suitable for:
- Reports
- Presentations
- Model performance analysis

---

## 🧠 Trained Model

| Attribute | Details |
|--------|--------|
Model File | `models/unet_oil_spill_model.h5` |
Architecture | U-Net |
Input Shape | 256 × 256 × 1 |
Task | Oil Spill Segmentation |

The trained model is saved separately for reuse in:
- Evaluation
- Visualization
- Future deployment

---

🚀 Module 6: Deployment Status

⚠️ Status: Attempted & Documented

A Streamlit-based web application (app.py) was developed to:

Upload SAR images

Run trained model inference

Display predicted masks and overlays

Deployment Challenges

TensorFlow / Keras version incompatibility

Legacy model serialization issues

To avoid modifying finalized training artifacts close to submission, deployment was documented but not finalized.

▶️ How to Run the Project (Local)
Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Run Streamlit App
python -m streamlit run app.py

Step 3: Test the Model

Upload a SAR image

View predicted oil spill segmentation



This module will be added in a future update.

---

🎯 Project Goals

Automate oil spill detection from satellite imagery

Reduce dependency on manual monitoring

Improve environmental surveillance efficiency

Demonstrate real-world deep learning application in remote sensing

🧠 Key Learnings

SAR image characteristics & challenges

Dice-based metrics for segmentation tasks

End-to-end ML pipeline design

Practical deployment constraints in ML systems
## 🛠️ Technology Stack

| Category | Tools |
|-------|------|
Programming Language | Python |
Deep Learning | TensorFlow, Keras |
Image Processing | OpenCV |
Data Handling | NumPy |
Visualization | Matplotlib |
Deployment (Planned) | Streamlit |
Version Control | Git & GitHub |

---

## 📌 Version Control Guidelines

- All development is performed on a **personal branch**
- The `main` branch remains protected
- No files are edited directly via GitHub UI
- All changes are committed and pushed from local setup

---

## 👤 Author

**Piyush Ranjan**  

🙏 Acknowledgments

Infosys Springboard Mentor Program

Project Mentor for guidance & review

Zenodo community for dataset resources

Open-source ML ecosystem

🎉 Final Notes

This project demonstrates a complete applied machine learning workflow, from satellite data analysis to model evaluation and visualization.

It highlights technical depth, engineering discipline, and real-world constraints, making it a strong and practical machine learning project.
 

---

## 📄 License

This project is developed for academic and internship purposes.

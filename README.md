# Oil Spill Detection using Image Segmentation

## Overview
Oil spills pose serious environmental and economic risks to marine and coastal ecosystems.  
This project focuses on detecting oil spill regions from satellite imagery using deep learning–based image segmentation.

A **U-Net–style Convolutional Neural Network (CNN)** is used to perform pixel-wise classification, identifying oil spill areas in water bodies.

---

## Problem Statement
Given satellite images of water surfaces, automatically identify and segment regions affected by oil spills to support early detection, monitoring, and environmental response efforts.

---

## Dataset
The dataset consists of paired **image–mask samples**:

- **Images**: Satellite images of water bodies  
- **Masks**: Binary segmentation masks indicating oil spill regions  
- **Splits**: Training and validation sets  

### Dataset Structure
images/
├── train/
└── val/

masks/
├── train/
└── val/


**Note:**  
Due to repository size constraints, the full dataset is not included in this repository.  
The code is fully reproducible when the dataset is placed in the expected directory structure.

---

## Project Workflow
1. Exploratory Data Analysis (EDA)
2. Image and mask visualization
3. Data preprocessing and normalization
4. Model building (U-Net architecture)
5. Training on real image–mask pairs
6. Prediction and qualitative evaluation
7. Model and result saving

---

## Exploratory Data Analysis (EDA)
EDA was performed to:
- Verify correct alignment between images and corresponding masks
- Inspect image dimensions and pixel distributions
- Visually validate segmentation targets

Sample image–mask visualizations confirm correct dataset pairing and integrity.

---

## Data Preprocessing
The following preprocessing steps were applied:
- Resize images and masks to a fixed size **(128 × 128)**
- Normalize pixel values to the range **[0, 1]**
- Expand mask dimensions for compatibility with CNN output
- Ensure binary masks for segmentation training

---

## Model Architecture
A **U-Net–style encoder–decoder CNN** was implemented with:
- Convolutional blocks with Batch Normalization
- Downsampling (encoder) for contextual feature extraction
- Upsampling (decoder) for spatial reconstruction
- Skip connections to preserve fine-grained spatial details

This architecture is well-suited for semantic segmentation tasks.

---

## Training Details
- **Loss Function**: Binary Cross-Entropy  
- **Evaluation Metric**: Dice Coefficient  
- **Optimizer**: Adam  

### Metric Selection
Accuracy is not a reliable metric for segmentation tasks due to class imbalance between background and oil spill pixels.  
The **Dice coefficient** better measures spatial overlap between predicted and ground truth masks.

---

## Results
The trained model successfully segments oil spill regions on validation images.  
Predicted masks show strong alignment with ground truth annotations, demonstrating effective spatial learning.

Qualitative results are saved as side-by-side visualizations:
- Original image
- Ground truth mask
- Predicted mask

---

## Model Saving
The trained model is saved in TensorFlow’s native format to enable:
- Reusability
- Inference
- Future deployment

Saved artifacts:
results/
├── model/
│ └── oil_spill_unet/
└── predictions/
├── sample_0.png
├── sample_1.png
└── sample_2.png



---

## Inference Application
A standalone inference script (`app.py`) is provided to:
- Load the trained model
- Run predictions on new images
- Save predicted masks and overlay visualizations

This demonstrates separation of **training** and **deployment-ready inference** logic.

---

## How to Run

### Install Dependencies
pip install -r requirements.txt


### Train the Model
Run the Jupyter notebook:
notebooks/oil_spill_detection.ipynb

### Run Inference
Outputs will be saved in:
results/app_output/



---

## Future Improvements
- Train on larger and more diverse satellite datasets
- Apply transfer learning for improved generalization
- Optimize model performance and inference speed
- Deploy as a web application using FastAPI or Streamlit
- Integrate real-time satellite data pipelines

---

## Conclusion
This project demonstrates an **end-to-end deep learning pipeline** for oil spill detection, covering data exploration, preprocessing, model development, evaluation, and deployment-oriented inference.

The approach can be extended to real-world environmental monitoring and decision-support systems.

---

## Author
Mohit Rohda




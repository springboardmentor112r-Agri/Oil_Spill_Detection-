🛢️ Oil Spill Detection Using Deep Learning on SAR Imagery

End-to-End Semantic Segmentation of Oil Spills from Satellite SAR Data

An applied deep learning project for environmental monitoring using Synthetic Aperture Radar (SAR) imagery.

📌 Project Overview

This repository contains an end-to-end implementation of an Oil Spill Detection system using Synthetic Aperture Radar (SAR) satellite imagery and deep learning–based semantic segmentation.

The project was developed as part of an internship program and follows a modular, milestone-driven approach, covering:

Data exploration & preprocessing

Model design & training

Evaluation & visualization

Deployment planning

🎯 Project Objectives

The primary goals of this project are:

Analyze SAR satellite imagery to detect oil spill regions

Preprocess SAR data and reduce noise artifacts

Design and train a deep learning segmentation model

Evaluate model performance using standard segmentation metrics

Visualize predicted oil spill regions effectively

Prepare the system for future real-time deployment

🛰️ Dataset Description
Attribute	Details
Dataset Name	Sentinel-1 SAR Oil Spill Dataset
Source	Zenodo
Image Type	Synthetic Aperture Radar (SAR)
Channels	Single-channel (Grayscale)
Classes	Oil Spill, Non-Spill, Look-Alike
Annotations	Pixel-level segmentation masks
Dataset Characteristics

The dataset includes real-world SAR images containing:

Confirmed oil spill regions

Clean sea surface (non-spill)

Look-alike patterns resembling oil spills

📂 Repository Structure
Oil_Spill_Detection/
│
├── notebooks/
│   └── Oil_Spill_Detection_End_to_End.ipynb
│
├── models/
│   └── unet_oil_spill_model.h5
│
├── screenshots/
│   ├── sample_input_image.png
│   ├── predicted_mask.png
│   └── overlay_visualization.png
│
├── app.py                  # Streamlit app (deployment module)
├── README.md
├── requirements.txt
└── .gitignore

📒 Notebook Overview

All project modules are implemented inside a single, well-structured notebook:

notebooks/Oil_Spill_Detection_End_to_End.ipynb


Modules are clearly separated using Markdown headings

Designed for easy review by mentors

Covers the complete ML lifecycle from data to results

🧩 Module-Wise Implementation
🔹 Module 2: Data Exploration & Preprocessing

This module focuses on understanding and preparing SAR data for model training.

Key Steps

Visualization of sample SAR images and masks

Statistical analysis of pixel intensity distributions

Image resizing to 256 × 256

Pixel normalization for stable training

SAR-specific speckle noise reduction

Data Augmentation Techniques

Horizontal & vertical flipping

Rotation

Brightness & contrast variation

🔹 Module 3: Model Development (Segmentation)
Component	Description
Model Type	U-Net
Task	Semantic Segmentation
Input	Single-channel SAR image
Output	Binary oil spill mask
Framework	TensorFlow / Keras

The U-Net architecture follows an encoder–decoder structure optimized for pixel-level prediction tasks.

🔹 Module 4: Training & Evaluation

Loss Functions

Dice Loss

Binary Cross-Entropy (BCE)

Evaluation Metrics

Accuracy

Intersection over Union (IoU)

Dice Coefficient

Precision

Recall

Hyperparameters were fine-tuned using validation performance to improve generalization.

🔹 Module 5: Visualization of Results

This module focuses on interpretability and result presentation.

Visual Outputs

Original SAR image

Ground truth segmentation mask

Predicted oil spill mask

Overlay visualization using color maps

These outputs are suitable for:

Technical reports

Presentations

Model performance analysis

🧠 Trained Model
Attribute	Details
Model File	models/unet_oil_spill_model.h5
Architecture	U-Net
Input Shape	256 × 256 × 1
Task	Oil Spill Segmentation

The trained model is saved separately for reuse in:

Evaluation

Visualization

Future deployment

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

⚠️ Deployment module will be finalized in a future update.

🛠️ Technology Stack
Category	Tools
Programming Language	Python
Deep Learning	TensorFlow, Keras
Image Processing	OpenCV
Data Handling	NumPy
Visualization	Matplotlib
Deployment (Planned)	Streamlit
Version Control	Git & GitHub
📌 Version Control Guidelines

All development performed on a personal branch

main branch remains protected

No direct edits via GitHub UI

All changes committed and pushed from local setup

🧠 Key Learnings

SAR image characteristics & challenges

Dice-based metrics for segmentation tasks

End-to-end ML pipeline design

Practical deployment constraints in ML systems

🎯 Project Goals

Automate oil spill detection from satellite imagery

Reduce dependency on manual monitoring

Improve environmental surveillance efficiency

Demonstrate real-world deep learning in remote sensing

👤 Author

Piyush Ranjan

🙏 Acknowledgments

Infosys Springboard Mentor Program

Project mentor for guidance & review

Zenodo community for dataset resources

Open-source ML ecosystem

📄 License

This project is developed for academic and internship purposes.

🎉 Final Notes

This project demonstrates a complete applied machine learning workflow, from satellite data analysis to model evaluation and visualization.

It reflects technical depth, engineering discipline, and real-world constraints, making it a strong and practical machine learning project.

🌊 Oil Spill Detection Using Deep Learning (U-Net)

📌 Project Overview
This project focuses on detecting oil spills in SAR (Synthetic Aperture Radar) satellite images using a deep learning–based image segmentation approach.
A U-Net architecture is implemented to accurately segment oil spill regions from SAR images, which is crucial for environmental monitoring and marine pollution control.

🎯 Objectives
•	Preprocess SAR images for oil spill detection
•	Reduce SAR image noise using speckle filtering
•	Perform data augmentation to improve model generalization
•	Train a U-Net segmentation model
•	Evaluate model performance using segmentation metrics
•	Visualize predictions and oil spill overlays

🧠 Model Architecture
•	Model: U-Net (Convolutional Neural Network for image segmentation)
•	Input size: 256 × 256 × 1
•	Optimizer: Adam
•	Loss Function: Dice Loss
•	Metrics Used:
o	Dice Coefficient
o	Accuracy
o	Precision
o	Recall



🧪 Preprocessing Steps
•	Convert SAR images to single-band grayscale
•	Resize images to 256 × 256
•	Normalize pixel values safely
•	Apply median-based speckle noise filtering
•	Expand dimensions for CNN compatibility

🔁 Data Augmentation
To improve robustness, the following augmentations are applied:
•	Horizontal flipping
•	Vertical flipping

⚙️ Training Configuration
•	Batch Size: 4
•	Epochs: 25
•	Steps per Epoch: Calculated dynamically based on dataset size
•	Data Generator: Custom Python generator for efficient training

📊 Model Evaluation
Training performance is visualized using:
•	Dice coefficient & accuracy curves
•	Precision & recall curves

🖼️ Results Visualization
The model outputs:
•	Original SAR image
•	Ground truth oil spill mask
•	Predicted oil spill segmentation
•	Overlay visualization showing detected oil spills in red

🛠️ Technologies & Libraries Used
•	Python
•	NumPy
•	OpenCV
•	Matplotlib
•	ImageIO
•	TensorFlow / Keras

🚀 How to Run
1.	Clone the repository
2.	Update dataset path in the notebook
3.	Install required dependencies
4.	Run the notebook step by step

📌 Key Highlights
•	Uses Dice Loss for better segmentation performance
•	Handles SAR image noise effectively
•	Custom data generator for memory efficiency
•	Suitable for real-world environmental monitoring applications


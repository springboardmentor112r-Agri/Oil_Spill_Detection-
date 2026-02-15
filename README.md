
Oil Spill Detection Using SAR Images and Deep Learning

Submitted by Sanskar Sutar

1. Introduction
Oil spills in oceans are a serious environmental problem and need to be detected as early as possible. One of the most reliable ways to detect oil spills is by using Synthetic Aperture Radar (SAR) images because they can capture images in all weather conditions and both day and night.
In this project, a deep learning based system is developed to classify SAR images into Oil Spill and No Oil Spill categories. The complete work includes image preprocessing, model training, testing, and deployment of the model so that it can be accessed publicly.

2. Objectives
The main objectives of this project are:
•	To preprocess SAR images and reduce noise
•	To train a deep learning model for oil spill detection
•	To test the model using unseen data
•	To deploy the trained model and generate a public link

3. Dataset Description
The dataset used in this project contains SAR images divided into two classes:
•	OilSpill
•	NoSpill
Each class is further divided into training and testing folders.
Dataset Structure:
Sanskar_Dataset/
 ├── OilSpill/
 │   ├── Train/
 │   └── Test/
 └── NoSpill/
     ├── Train/
     └── Test/

4. Image Preprocessing
SAR images usually contain speckle noise, which can affect the performance of the model. To improve image quality, preprocessing was performed before training.
The following steps were applied:
•	Conversion of images to grayscale
•	Noise removal using Non-Local Means Denoising
•	Contrast enhancement using CLAHE
•	Light Gaussian blur for smoothing
After preprocessing, the images were saved in a separate folder so that the original dataset remains unchanged.
Preprocessed Dataset Structure:
Sanskar_dataset_preprocessed/
 ├── OilSpill/
 │   ├── Train/
 │   └── Test/
 └── NoSpill/
     ├── Train/
     └── Test/

5. Model Training
The deep learning model was trained using the preprocessed training images. During training, the model learned the visual differences between oil spill and non-oil spill regions.
The training process included:
•	Loading the dataset
•	Compiling the model
•	Training the model on training data
•	Saving the trained model

6. Model Testing and Evaluation
After training, the model was tested using the test dataset. This helped in checking how well the model performs on new and unseen images. The accuracy obtained shows that the model is able to correctly classify oil spill and non-oil spill images.


7. Deployment
The trained model was deployed as a web application so that it can be used by anyone.
The deployed system allows:
•	Uploading a SAR image
•	Getting instant prediction (Oil Spill or No Oil Spill)
Images attched:-
1)<img width="1327" height="949" alt="image" src="https://github.com/user-attachments/assets/4f6ef8ee-c664-4ea1-a525-37fc27a0e512" />
2)<img width="1272" height="941" alt="image" src="https://github.com/user-attachments/assets/b890fa68-ce64-428a-a44a-95601aa21690" />



Deployment Link:
Oil Spill Detection · Streamlit

8. Tools and Technologies Used
•	Python
•	OpenCV
•	NumPy
•	Matplotlib
•	TensorFlow / Keras
•	Google Colab
•	GitHub

9. Conclusion
In this project, a deep learning based oil spill detection system was successfully developed using SAR images. Image preprocessing played an important role in improving model performance. The deployed model provides a simple and effective way to identify oil spills from SAR images.

10. Future Improvements
•	Use a larger dataset for better accuracy
•	Apply advanced CNN models
•	Perform segmentation instead of only classification
•	Deploy the system on cloud platforms



Deployment Link-
https://sanskar-19-sar-oill-spill-detection.streamlit.app/

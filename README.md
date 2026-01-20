<h1 align="center"> <strong>🛢️ Oil Spill Detection Using Deep Learning</strong> </h1>
An end-to-end deep learning system for automated oil spill detection from satellite imagery.
🌍 Overview

Oil spills pose a serious threat to marine ecosystems, biodiversity, and coastal economies. Traditional monitoring methods rely heavily on manual inspection of satellite images, which is slow, labor-intensive, and prone to human error.

This project presents a deep learning–based image classification system that automatically detects oil spills from satellite imagery. By leveraging convolutional neural networks (CNNs), the system distinguishes between oil spill and non-oil spill regions, enabling faster and more reliable environmental monitoring.

🎯 Project Goals
<ul> <li>Automatically detect oil spills from satellite images</li> <li>Reduce dependency on manual image inspection</li> <li>Enable early detection for environmental protection</li> <li>Develop a complete AI pipeline from data analysis to deployment</li> </ul>
🧠 Problem Definition

Conventional oil spill detection approaches are often time-consuming, inconsistent, and difficult to scale. The growing availability of satellite imagery demands an intelligent system capable of analyzing large volumes of data efficiently.

The key challenge addressed in this project is to design an automated, accurate, and scalable deep learning solution that can classify satellite images and reliably identify oil spill regions.

📊 Dataset Summary
<ul> <li><b>Source:</b> Satellite imagery dataset</li> <li><b>Classes:</b> Oil Spill, Non-Oil Spill</li> <li><b>Data Type:</b> RGB images (JPEG / PNG)</li> <li><b>Task:</b> Binary image classification</li> </ul>
📁 Dataset Organization
<pre>
Oil_Spill_Detection/
│
├── AI_Driven_System_for_Oil_Spill_Identification_and_Monitoring.ipynb
│   ├── End-to-end model training
│   ├── Segmentation evaluation
│   └── Result visualization
│
├── AI_Driven_System_for_Oil_Spill_Identification_and_Monitoring_(EDA).ipynb
│   ├── Exploratory Data Analysis (EDA)
│   ├── Dataset inspection & statistics
│   └── Data insights for preprocessing
│
├── unet_final_industry.keras
│   └── Trained U-Net segmentation model (GPU-trained)
│
├── dataset.zip
│   └── Raw satellite images and ground-truth segmentation masks
│
├── preprocess.zip
│   └── Preprocessed and cleaned data used for training
│
├── README.md
│   └── Project documentation
│
├── LICENSE
│   └── MIT License
│
└── .gitattributes
    └── Git configuration for large files
</pre>

🔄 System Workflow
<pre>
Oil_Spill_Detection_Pipeline/
│
├── 01 Data Collection
│   └── Raw satellite imagery acquisition
│
├── 02 Exploratory Data Analysis (EDA)
│   └── Dataset inspection, visualization, and statistics
│
├── 03 Data Preprocessing
│   └── Image resizing, normalization, and augmentation
│
├── 04 CNN Model Design
│   └── CNN architecture definition
│
├── 05 Model Training
│   └── Model training and validation
│
├── 06 Model Evaluation
│   └── Performance metrics and analysis
│
├── 07 Model Persistence
│   └── Saving trained model artifacts
│
└── 08 Deployment
    └── Model deployment (Streamlit / Flask)
</pre>

🔍 Exploratory Data Analysis (EDA)

EDA was performed to gain insights into the dataset and guide model development.

<ul> <li>Dataset size and structure analysis</li> <li>Class distribution and imbalance check</li> <li>Visualization of sample images</li> <li>Image resolution and quality assessment</li> </ul>

📓 <b>Notebook:</b> EDA_and_preprocessing_oil_spill_9.ipynb

🧹 Data Preprocessing

To prepare the data for training, the following preprocessing steps were applied:

<ul> <li>Loading images from directory structure</li> <li>Resizing images to a fixed input size</li> <li>Normalizing pixel values (0–255 → 0–1)</li> <li>Encoding class labels</li> <li>Splitting data into training and testing sets</li> <li>Optional data augmentation for robustness</li> </ul>
🤖 Model Design
<ul> <li><b>Model Type:</b> Convolutional Neural Network (CNN)</li> <li><b>Framework:</b> TensorFlow / Keras</li> </ul>
Architecture Components
<ul> <li>Convolutional layers for feature extraction</li> <li>MaxPooling layers for spatial reduction</li> <li>Flatten layer</li> <li>Fully connected (Dense) layers</li> <li>Sigmoid output layer for binary classification</li> </ul>
🏋️ Model Training
<ul> <li><b>Loss Function:</b> Binary Cross-Entropy</li> <li><b>Optimizer:</b> Adam</li> <li><b>Evaluation Metric:</b> Accuracy</li> <li><b>Epochs & Batch Size:</b> Tuned experimentally</li> </ul>
📈 Model Evaluation

The trained model was evaluated using:

<ul> <li>Overall classification accuracy</li> <li>Training vs validation loss curves</li> <li>Training vs validation accuracy curves</li> </ul>

✅ Performance Result
<p> <b>Final Accuracy:</b> ~82% </p>

This performance is suitable for academic evaluation and prototype-level deployment.

<h2>📊 Results & Visualizations</h2>

<p align="center">
  <img
    src="[RESULT_IMAGE_1]"
    alt="Oil Spill Detection – Image"
    width="48%"
  />
  <img
    src="[RESULT_IMAGE_2]"
    alt="Oil Spill Detection – Image"
    width="48%"
  />
</p>

<p align="center">
  <em>
    Qualitative results showing model predictions on satellite imagery.
  </em>
</p>


💾 Model Persistence

The trained model is saved for reuse and deployment:

<pre>model.save("oil_spill_model.keras")</pre>

🌐 Deployment Strategy
Deployable Link: https://oil-spill-detection-app-dbzewqefbtj6bpvujylg9j.streamlit.app/

<p align="center">
  <a href="https://oil-spill-detection-app-dbzewqefbtj6bpvujylg9j.streamlit.app/" target="_blank">
    <img
      src="https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-success?style=for-the-badge"
      alt="Live Demo Link"
    />
  </a>
</p>

The model can be deployed using modern web frameworks for real-time inference.

<ul> <li><b>Streamlit</b> – Interactive web application</li> <li><b>Flask</b> – REST API–based deployment</li> <li><b>Cloud Platforms</b> – Optional scalable hosting</li> </ul>
Deployment Features
<ul> <li>Upload satellite images</li> <li>Instant prediction: Oil Spill / Non-Oil Spill</li> <li>User-friendly interface</li> </ul>
🛠️ Technology Stack
<ul> <li><b>Language:</b> Python</li> <li><b>Environment:</b> Google Colab / Jupyter Notebook</li> <li><b>Libraries:</b> NumPy, Pandas, Matplotlib, Seaborn</li> <li><b>Computer Vision:</b> OpenCV / PIL</li> <li><b>ML Framework:</b> TensorFlow / Keras</li> <li><b>Deployment:</b> Streamlit / Flask</li> </ul>

📂 Project Layout
<pre>
Oil_Spill_Detection/
│
├── EDA_and_preprocessing_oil_spill_9.ipynb
│   ├── Dataset exploration and visualization
│   ├── Class distribution analysis
│   └── Image preprocessing and normalization
│
├── model_training.ipynb
│   ├── CNN architecture definition
│   ├── Model training and validation
│   └── Performance evaluation and accuracy metrics
│
├── oil_spill_model.keras
│   └── Trained CNN binary classification model
│
├── dataset/
│   ├── Oil_Spill/
│   │   └── Satellite images containing oil spills
│   │
│   └── Non_Oil_Spill/
│       └── Satellite images without oil spills
│
├── app.py
│   └── Streamlit / Flask application for inference
│
├── requirements.txt
│   └── Python dependencies for training and deployment
│
├── README.md
│   └── Project documentation
│
└── LICENSE
    └── MIT License
</pre>


✅ Key Advantages
<ul> <li>Fast and automated oil spill detection</li> <li>Reduces human effort and monitoring time</li> <li>Reusable and scalable pipeline</li> <li>Potential integration with satellite monitoring systems</li> </ul>
⚠️ Limitations
<ul> <li>Model performance depends on image quality</li> <li>Requires larger datasets for improved accuracy</li> <li>Environmental noise may affect predictions</li> </ul>
🔮 Future Scope
<ul> <li>Apply transfer learning (ResNet, VGG, EfficientNet)</li> <li>Expand dataset for higher generalization</li> <li>Integrate real-time satellite feeds</li> <li>Develop a mobile-based interface</li> </ul>
👩‍💻 Author

<b>Kalyani Patil</b>

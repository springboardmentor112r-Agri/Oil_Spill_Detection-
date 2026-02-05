🌊 AI-Driven Oil Spill Detection & Monitoring System
<p align="center">
  <img
    src="https://res.cloudinary.com/dbiiwike0/image/upload/v1768200995/ChatGPT_Image_Jan_12_2026_11_45_33_AM_zeyqar.png"
    alt="OilSpill AI – SAR-Based Oil Spill Detection"
    width="100%"
  />
</p>




 <strong>Deep Learning–based semantic segmentation system for detecting oil spills in satellite imagery using U-Net.</strong> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue"/> <img src="https://img.shields.io/badge/Framework-TensorFlow-orange"/> <img src="https://img.shields.io/badge/Model-U--Net-success"/> <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow"/> <img src="https://img.shields.io/badge/License-MIT-green"/> <p align="center"> <img src="https://img.shields.io/badge/Hugging%20Face-🤗-yellow"/></p>
</p>

📌 Project Overview

Oil spills pose a severe threat to marine ecosystems, coastal economies, and global environmental health. Manual monitoring and traditional detection methods are slow, expensive, and error-prone.

This project presents an AI-driven oil spill identification and monitoring system that leverages deep learning–based semantic segmentation to automatically detect oil spill regions from satellite imagery with high precision.

Using a U-Net architecture, the system learns pixel-level patterns of oil spill regions, enabling accurate segmentation even in complex oceanic backgrounds.

🎯 Key Objectives
<ul> <li>Automate oil spill detection from satellite images</li> <li>Perform <b>pixel-level segmentation</b> instead of coarse classification</li> <li>Achieve high accuracy suitable for <b>industrial & environmental monitoring</b></li> <li>Provide <b>visual and quantitative evaluation</b> of predictions</li> </ul>
🧠 Model Architecture
<ul> <li><b>Model:</b> U-Net (Encoder–Decoder CNN)</li> <li><b>Task:</b> Binary semantic segmentation (Oil Spill vs Background)</li> <li><b>Input:</b> Satellite imagery</li> <li><b>Output:</b> Binary segmentation mask</li> </ul>

<b>Why U-Net?</b>

<ul> <li>Strong spatial localization</li> <li>Skip connections preserving fine-grained features</li> <li>Proven performance in environmental and medical segmentation tasks</li> </ul>

🗂️ Repository Structure
<pre>
Oil_Spill_Detection/
│
├── AI_Driven_System_for_Oil_Spill_Identification_and_Monitoring.ipynb
│   ├── End-to-end training
│   ├── Model evaluation
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

🧩 Problem Statement
📌 Overview

Oil spills pose a major threat to marine ecosystems, coastal regions, and local economies, making early detection essential. Although satellite imagery enables large-scale ocean monitoring, traditional oil spill detection methods rely on manual inspection and physical surveillance, which are slow, labor-intensive, and difficult to scale.

Additionally, oil spills often exhibit subtle visual patterns and low contrast against surrounding water, limiting the effectiveness of classical image processing techniques. These challenges highlight the need for an automated, accurate, and scalable AI-driven system capable of detecting and precisely localizing oil spills from satellite imagery.

📍 Key Challenges
<ul> <li>Manual inspection of satellite images is <b>time-consuming and non-scalable</b></li> <li>Delayed detection leads to <b>greater environmental damage</b></li> <li>Oil spills exhibit <b>low contrast and irregular shapes</b></li> <li>Classical methods lack <b>robustness and generalization</b></li> <li>Large satellite datasets require <b>automated analysis</b></li> <li>Accurate <b>pixel-level localization</b> is critical for response planning</li> </ul>
🎯 Problem to Solve
<p> <b>How can oil spills be automatically and accurately detected from satellite imagery at scale, while precisely localizing affected regions and minimizing human intervention?</b> </p>
💡 Proposed Solution – AI-Driven Oil Spill Detection System
📌 Solution Overview

We developed an AI-driven oil spill identification and monitoring system using deep learning–based semantic segmentation. The system employs a U-Net architecture to learn spatial and visual features of oil-contaminated regions directly from satellite images.

Rather than performing image-level classification, the model generates pixel-level segmentation masks, enabling precise localization of oil spills and scalable real-world deployment.

⚙️ How the Solution Works
<ul> <li>Automatically analyzes satellite imagery</li> <li>Uses a <b>U-Net encoder–decoder architecture</b> for fine-grained segmentation</li> <li>Trained with preprocessing, augmentation, and GPU acceleration</li> <li>Evaluated using <b>Accuracy, IoU, Dice, Precision, and Recall</b></li> <li>Visualizes predictions alongside ground truth for interpretability</li> </ul>
🚀 Impact & Benefits
<ul> <li>Enables <b>early oil spill detection</b> and rapid response</li> <li>Reduces reliance on <b>manual monitoring</b></li> <li>Scales efficiently to <b>large satellite datasets</b></li> <li>Improves <b>decision-making for environmental agencies</b></li> <li>Designed to be <b>deployment-ready</b> via web interfaces</li> </ul>
📓 Notebooks Explained
1️⃣ AI_Driven_System_for_Oil_Spill_Identification_and_Monitoring.ipynb

<b>(Training & Visualization Notebook)</b>

<ul> <li>Dataset loading & preparation</li> <li>Image and mask preprocessing</li> <li>Model architecture definition (U-Net)</li> <li>Model training on GPU (Google Colab)</li> <li>Validation and performance evaluation</li> <li>Visualization of predictions</li> <li>Saving the trained model</li> </ul>

<b>Outputs:</b>

<ul> <li>Trained segmentation model</li> <li>Qualitative visual results</li> <li>Performance metrics</li> </ul>
2️⃣ AI_Driven_System_for_Oil_Spill_Identification_and_Monitoring_(EDA).ipynb

<b>(Exploratory Data Analysis Notebook)</b>

<ul> <li>Dataset inspection and statistics</li> <li>Image and mask visualization</li> <li>Distribution analysis</li> <li>Data quality checks</li> <li>Insights for preprocessing and model design</li> </ul>
📦 Dataset
<ul> <li><b>dataset.zip</b> – Raw satellite images which are pre-process and used for training of model</li> <li><b>preprocess.zip</b> – Preprocessed and cleaned training data</li> </ul> <p><b>Note:</b> Large files are recommended to be hosted externally (e.g., Hugging Face).</p>
💾 Trained Model
<ul> <li><b>unet_final_industry.keras</b> – GPU-trained U-Net segmentation model</li> </ul>

<b>Training Highlights:</b>

<ul> <li>Loss Function: Dice Coefficient : 0.72</li> <li>Metrics: Accuracy : 88.2%,  IoU : 0.56,  Recall : 0.61 </li> <li>Hardware: Google Colab GPU</li> </ul>

<h2>📊 Results & Visualizations</h2>

<p align="center">
    <img
    src="https://res.cloudinary.com/dbiiwike0/image/upload/v1768200101/Screenshot_2026-01-12_114611_jidvta.png"
    alt="Oil Spill Detection Result 2"
    width="48%"
  />
    <img
    src="https://res.cloudinary.com/dbiiwike0/image/upload/v1768200108/Screenshot_2026-01-12_121127_pe0fk7.png"
    alt="Oil Spill Detection Result 1"
    width="48%"
  />
</p>

<p align="center">
  <em>Left: Model prediction oil spill region · Right: Segmentation output highlighting no oil spill regions</em>
</p>

🧰 Tech Stack
<p align="left"> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="40"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40"/> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" width="40"/> <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="42"/>
    
🚀 Deployable Link

<h2>🚀 Live Demo & Deployment</h2>

<p align="center">
  <strong>Experience the AI-powered oil spill detection system in action.</strong><br/>
  Upload satellite imagery and view real-time segmentation results powered by deep learning.
</p>

Deployable Link: https://huggingface.co/spaces/ritwik01/oil_spill_analysis

<p align="center">
<a href="https://huggingface.co/spaces/ritwik01/oil_spill_analysis" target="_blank">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow?style=for-the-badge"/>
  </a>
  <a href="https://github.com/YOUR_GITHUB_REPO" target="_blank">
    <img src="https://img.shields.io/badge/Star-GitHub-black?style=for-the-badge"/>
  </a>
</p>


🔮 Future Improvements
<ul> <li>Deploy as a real-time monitoring API</li> <li>Integrate temporal satellite data for spill tracking</li> <li>Improve generalization with multi-sensor data</li> <li>Optimize for edge and low-resource environments</li> </ul>
👨‍💻 Author

<b>Ritwik</b><br/>
AI / Machine Learning Enthusiast Love to make project
Focused on real-world, industry-grade deep learning systems

📜 License

This project is licensed under the <b>MIT License</b>.

<h1 align="center">⭐ If you like this project, give it a star — it really helps!</h1>

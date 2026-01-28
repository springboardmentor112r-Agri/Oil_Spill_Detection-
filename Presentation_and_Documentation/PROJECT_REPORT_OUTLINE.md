# Project Report: AI-Driven Oil Spill Detection using U-Net and SAR Images

## 1. Abstract
-   Concise summary of the project, including the problem, methodology, key results, and main conclusions.
-   Highlight the significance of the work for oil spill detection.

## 2. Introduction
### 2.1 Problem Statement
-   Background on oil spills: environmental impact, economic consequences, and detection challenges.
-   Role of SAR imagery in oil spill detection.
### 2.2 Project Goals
-   Clearly state the primary objectives of the project (e.g., develop a U-Net model for segmentation, deploy a Streamlit application).
-   List specific deliverables.
### 2.3 Significance
-   Explain the importance of this project to the field of environmental monitoring and disaster response.
-   Discuss the potential impact of an accurate and accessible detection system.

## 3. Related Work
-   Overview of traditional oil spill detection methods.
-   Review of existing AI/ML approaches for oil spill detection (e.g., traditional image processing, other deep learning models).
-   Comparison of different SAR filtering techniques and their application in this domain.
-   Positioning of this project within the current literature, highlighting unique contributions or improvements.

## 4. Dataset and Preprocessing
### 4.1 Data Sources
-   Description of the SAR (PALSAR, Sentinel) dataset used, including image types and characteristics.
-   Mention data acquisition from Kaggle and Google Drive integration.
### 4.2 Initial Alignment and Cleanup
-   Details on initial data inspection (e.g., `os.walk` output).
-   Explanation of image-mask misalignment issues encountered (e.g., `(1).png` files, masks without images).
-   Process for removing misaligned mask files from training and validation sets.
-   Final aligned counts for training and validation sets.
### 4.3 Dataset Structuring
-   Process of splitting the initial training data into new training and test sets.
-   Final sizes of the training, validation, and test sets after splitting.
### 4.4 Resizing
-   Target resolution: (128, 128) pixels.
-   Interpolation methods used: `PIL.Image.LANCZOS` for images, `PIL.Image.NEAREST` for masks, and justification.
-   Creation of `dataset_resized_128x128` directory structure.
### 4.5 Normalization and SAR Filtering
-   Normalization of pixel values to [0, 1] range.
-   Application of SAR-specific speckle noise reduction filter (`skimage.restoration.denoise_wavelet`).
-   Parameters used for filtering (wavelet, mode, sigma).
-   Creation of `dataset_processed_128x128` directory structure.
### 4.6 Data Augmentation
-   Rationale for data augmentation (increasing dataset size, improving generalization).
-   Techniques applied: HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast.
-   Parameters and probabilities for each augmentation.
-   Impact on training dataset size.
-   Creation of `dataset_augmented_128x128` directory structure.
### 4.7 Final Dataset Splits
-   Tabular summary of the final number of image-mask pairs in training, validation, and test sets.

## 5. Methodology
### 5.1 U-Net Model Architecture
-   Detailed description of the U-Net architecture.
-   Input layer specification (128x128x1).
-   Encoder path: number of blocks, filter sizes (32, 64, 128), convolutional layers, activation functions, pooling layers.
-   Bottleneck: filter size (256).
-   Decoder path: Conv2DTranspose layers, skip connections, filter sizes (128, 64, 32), convolutional layers, activation functions.
-   Output layer: 1x1 Conv2D with sigmoid activation.
-   Rationale for reduced filter sizes and `TARGET_SIZE`.
### 5.2 Training Setup
-   **Optimizer**: Adam, with initial learning rate (e.g., 1e-4).
-   **Loss Function**: Binary Cross-Entropy, justification for choice.
-   **Metrics**: Mean IoU, Binary Accuracy, Precision, Recall.
-   **Callbacks**: EarlyStopping (monitor, patience, restore_best_weights), ReduceLROnPlateau (monitor, factor, patience, min_lr).
-   **Mixed Precision Training**: Implementation details (`tf.keras.mixed_precision.set_global_policy('mixed_float16')`) and benefits.

## 6. Results
### 6.1 Quantitative Evaluation
-   **Test Set Metrics**: Present the final loss, Mean IoU, Binary Accuracy, Precision, and Recall on the unseen test set.
-   **Training History Analysis**: Discuss trends in loss and metrics (training vs. validation) over epochs.
### 6.2 Qualitative Evaluation
-   Visual examples of predictions from the test set.
-   Side-by-side comparison of original image, ground truth mask, predicted mask, and overlaid prediction.
-   Discussion of specific examples: areas of good performance, and instances where the model struggles (e.g., fine details, fragmented spills).

## 7. Discussion
-   Interpretation of the quantitative results: What do the metrics tell us about the model's performance?
-   Strengths of the model: Robustness, ability to generalize, effectiveness of preprocessing.
-   Limitations: Performance on challenging cases, potential false positives/negatives, boundary precision.
-   Challenges encountered during development (e.g., library incompatibility, resource constraints, data management).

## 8. Conclusion
-   Summarize the main findings and achievements of the project.
-   Reiterate the project's contribution to oil spill detection.

## 9. Future Work
-   Suggestions for improving model performance (e.g., advanced architectures, more sophisticated augmentation, different loss functions).
-   Ideas for expanding the dataset or incorporating multi-modal data.
-   Enhancements for the Streamlit application (e.g., real-time processing, different visualization options, user feedback).
-   Exploration of deployment to cloud platforms for wider accessibility.

## 10. References
-   List all cited academic papers, datasets, and significant resources.

## 11. Appendices
-   Code Snippets (e.g., for key preprocessing steps, model definition, training loop).
-   Additional Visualizations (e.g., more prediction examples, training curves).
-   Full Model Summary (text output).
-   Detailed environment setup information.

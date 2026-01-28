# Oil Spill Detection using U-Net and SAR Images

## Project Overview
This project implements an AI-driven system for detecting oil spills in Synthetic Aperture Radar (SAR) images using a U-Net convolutional neural network. The system is designed to identify oil spill regions, classify images as 'Oil Spill Detected' or 'No Oil Spill Detected', and visualize the results. The application is deployed as a user-friendly web interface using Streamlit.

### Key Features:
-   **U-Net Segmentation Model**: A customized U-Net architecture optimized for SAR image segmentation.
-   **Data Preprocessing**: Images are resized, normalized, and processed with SAR-specific speckle noise reduction filters.
-   **Data Augmentation**: Training data is augmented with techniques like flipping, rotating, and brightness/contrast adjustments to improve model generalization.
-   **Streamlit Web Application**: An interactive web interface for uploading SAR images, obtaining real-time predictions, and visualizing segmentation masks.
-   **Classification**: Classifies images as 'Oil Spill Detected' or 'No Oil Spill Detected' based on the predicted mask.
-   **Visualization**: Displays original images, predicted segmentation masks, and overlaid predictions with transparency.
-   **Download Options**: Allows users to download the predicted mask and the overlaid image.

## Setup Instructions
This project is designed to be run in a Google Colab environment, leveraging its GPU resources for model training and inference. Follow these steps to set up your environment.

### 1. Mount Google Drive
To access the dataset and save model artifacts, mount your Google Drive at the beginning of your Colab session:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Python Libraries
Install all necessary Python libraries. Ensure `albumentations` and `pyngrok` are installed for data augmentation and exposing the Streamlit app, respectively.

```bash
!pip install tensorflow numpy pillow albumentations scikit-image streamlit pyngrok opencv-python matplotlib
```

### 3. Obtain and Organize Dataset
Ensure your dataset (named `Deep SAR (SOS) Dataset`) is placed in your Google Drive at `MyDrive/Deep SAR (SOS) Dataset/`. The expected structure is:
```
Deep SAR (SOS) Dataset/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── masks/
│       ├── masks/
│       │   ├── train/
│       │   └── val/
│       └── test/ (created during preprocessing)
├── dataset_resized_128x128/ (created during preprocessing)
├── dataset_processed_128x128/ (created during preprocessing)
├── dataset_augmented_128x128/ (created during preprocessing)
└── unet_oil_spill_segmentation_model_128x128.keras (saved after training)
```

### 4. Configure Ngrok Authentication Token
`ngrok` is used to create a public URL for your Streamlit application. You need an authentication token, which can be obtained from your ngrok dashboard (https://dashboard.ngrok.com/get-started/your-authtoken). Replace `"YOUR_NGROK_AUTH_TOKEN"` with your actual token.

```python
# In a Colab cell or terminal
!ngrok authtoken "YOUR_NGROK_AUTH_TOKEN"
```

## Usage Guidelines

### 1. Preprocess Data and Train Model
Before launching the Streamlit app, you must run the data preprocessing (resizing, normalization, filtering, augmentation) and model training cells in the Colab notebook. These steps generate the processed datasets and the `unet_oil_spill_segmentation_model_128x128.keras` model file, which are essential for the Streamlit app.

### 2. Launch the Streamlit Application
Once preprocessing and training are complete, and `app.py` is saved in your `Deep SAR (SOS) Dataset` directory, launch the Streamlit application from a Colab terminal:

**a. Open a New Terminal in Colab**:
   - Click on 'Runtime' > 'Change runtime type' and ensure 'GPU' is selected. Then click 'Save'.
   - Click on 'Terminal' in the left sidebar and select 'New terminal'.

**b. Navigate to the Application Directory**:
   ```bash
   cd /content/drive/MyDrive/Deep SAR (SOS) Dataset/
   ```

**c. Stop any existing Streamlit or Ngrok processes**:
   ```bash
   !pkill -f streamlit
   !pkill -f ngrok
   ```

**d. Launch the Streamlit app with Ngrok**:
   Execute the following Python code in a **Colab code cell** (not the terminal) or a separate Python script, after ensuring `pyngrok` is installed and `ngrok` is authenticated:
   ```python
   from pyngrok import ngrok

   # Terminate any existing ngrok tunnels
   ngrok.kill()

   # Start a new ngrok tunnel for Streamlit on port 8501
   public_url = ngrok.connect(8501)

   print(f"Streamlit App URL: {public_url}")

   # Run Streamlit in the background
   !streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false &
   ```

**e. Access the App**: After executing the above, `ngrok` will provide a public URL. Click on this URL to open the Streamlit application in your web browser.

### 3. Interact with the Application
-   **Upload Image**: Use the 'Upload Satellite Image' button to select a SAR image (PNG, JPG, JPEG format).
-   **View Original Image**: The uploaded image will be displayed.
-   **Detection Results**: After processing, the app will display:
    -   A classification: 'Oil Spill Detected' or 'No Oil Spill Detected'.
    -   The 'Predicted Segmentation Mask', showing the identified spill regions in white.
    -   An 'Overlay Visualization', where the predicted mask is overlaid in green with transparency on the original image.
-   **Download Options**: Buttons are provided to download the 'Predicted Mask' and the 'Overlay Image' for further analysis or reporting.

## Model Details

The U-Net model is configured with reduced filter sizes (32-64-128 in encoder/decoder) for a 128x128 input. It uses `Adam` optimizer, `BinaryCrossentropy` loss, and `MeanIoU`, `BinaryAccuracy`, `Precision`, `Recall` metrics. Training incorporates `EarlyStopping` and `ReduceLROnPlateau` callbacks for robust learning, along with mixed precision training for efficiency.

## Data Preprocessing Summary

-   **Image and Mask Alignment**: Ensures a 1:1 correspondence between images and masks across all datasets.
-   **Resizing**: All images and masks are resized to (128, 128) pixels.
-   **Normalization and Filtering**: Image pixel values are normalized to [0, 1] and a SAR-specific speckle noise reduction filter is applied.
-   **Data Augmentation**: Training data is augmented with horizontal/vertical flips, rotations, and brightness/contrast adjustments.

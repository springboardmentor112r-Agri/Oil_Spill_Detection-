# Oil Spill Detection System

AI-powered oil spill detection from satellite imagery using deep learning. This project implements an Attention U-Net architecture for semantic segmentation of oil spills in SAR (Synthetic Aperture Radar) satellite images.

## Features

- **Attention U-Net Architecture**: State-of-the-art segmentation model with attention mechanisms
- **Binary Segmentation**: Precise oil spill vs background classification
- **Real-time Inference**: Streamlit web application for interactive predictions
- **Area Estimation**: Calculates oil spill area in km² from satellite imagery
- **Comprehensive Metrics**: Dice Score, IoU, Precision, Recall evaluation
- **Easy Deployment**: Simple web interface for uploading and analyzing images

## Model Performance

- **Dice Coefficient**: ~0.70
- **IoU (Intersection over Union)**: ~0.57
- **Precision & Recall**: Balanced performance on test set
- Comparable to professional oil spill detection systems

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Oil_Spill_UNet_Model_Implementation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Download Dataset

1. Download the Oil Spill Detection dataset from Kaggle:
   - Dataset: [Oil Spill Detection](https://www.kaggle.com/datasets/...)
   - Or use any SAR satellite imagery with corresponding masks

2. Place the raw dataset in `data/raw/` folder:
```
data/raw/
├── images/    # Satellite images (grayscale/SAR)
└── masks/     # Binary segmentation masks
```

### Organize Dataset

Split the dataset into train/validation/test sets (70/15/15):

```bash
python organize_data.py
```

This creates:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Usage

### 1. Test Dataset Loading

Verify dataset is properly loaded:

```bash
python dataset.py
```

### 2. Test Model Architecture

Check model is correctly initialized:

```bash
python model.py
```

### 3. Train Model

Train the Attention U-Net model:

```bash
python train.py
```

**Training Configuration:**
- Epochs: 20
- Batch Size: 8
- Learning Rate: 1e-4
- Loss Function: Dice Loss
- Optimizer: Adam

**Output:**
- `best_model.pth` - Best model weights (lowest validation loss)
- `training_history.png` - Training/validation loss curves

### 4. Evaluate Model

Calculate comprehensive metrics on test set:

```bash
python evaluate.py
```

**Metrics Calculated:**
- Dice Coefficient (F1 Score)
- IoU (Jaccard Index)
- Precision
- Recall
- Distribution plots

**Output:**
- `evaluation_results.png` - Metrics distribution visualizations
- Console output with detailed statistics

### 5. Deploy Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

**Features:**
- Upload satellite images (PNG, JPG, TIFF)
- Adjust detection threshold
- View detection results with overlay
- Calculate oil spill area in km²
- Download binary masks and probability maps
- Interactive probability heatmap

Access the app at: `http://localhost:8501`

## Model Architecture

### Attention U-Net

The model uses an encoder-decoder architecture with attention gates:

**Encoder (Downsampling):**
- 4 convolutional blocks with max pooling
- Channels: 64 → 128 → 256 → 512

**Bottleneck:**
- Dense feature extraction (1024 channels)

**Decoder (Upsampling):**
- 4 transposed convolutional blocks
- Attention gates at each skip connection
- Channels: 512 → 256 → 128 → 64

**Output:**
- Single channel binary segmentation mask
- Sigmoid activation for probability

**Key Features:**
- Attention mechanisms focus on relevant features
- Skip connections preserve spatial information
- Handles class imbalance (oil pixels << background)

## Data Preprocessing

Following a simple and effective pipeline:

1. **Grayscale Conversion**: SAR images are single-channel
2. **Resize**: Standardize to 256×256 pixels
3. **Median Blur**: Reduce speckle noise (kernel size: 3)
4. **Normalization**: Scale pixel values to [0, 1]
5. **Mask Binarization**: Ensure masks are binary (0 or 1)

## Training Details

### Loss Function: Dice Loss

```python
Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
Loss = 1 - Dice
```

**Advantages:**
- Handles severe class imbalance
- Differentiable for backpropagation
- Directly optimizes segmentation metric

### Optimization

- **Optimizer**: Adam with default parameters
- **Learning Rate**: 1e-4 (fixed)
- **Batch Size**: 8
- **Epochs**: 20

### Data Augmentation

Currently using simple preprocessing. For improved performance, consider adding:
- Random horizontal/vertical flips
- Random rotations
- Brightness/contrast adjustments

## Project Structure

```
oil-spill-project/
├── data/                      # Dataset directory
│   ├── raw/                   # Raw downloaded data
│   ├── train/                 # Training set
│   ├── val/                   # Validation set
│   └── test/                  # Test set
├── model.py                   # Attention U-Net architecture
├── dataset.py                 # Data loader and preprocessing
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── app.py                     # Streamlit web application
├── organize_data.py           # Dataset splitting script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── best_model.pth            # Trained model weights (after training)
├── training_history.png      # Training curves (after training)
└── evaluation_results.png    # Evaluation plots (after evaluation)
```

## Results Interpretation

### Dice Coefficient
- **Range**: 0 to 1
- **Interpretation**:
  - 0.70+: Excellent segmentation
  - 0.60-0.70: Good segmentation
  - 0.50-0.60: Moderate segmentation
  - <0.50: Poor segmentation

### IoU (Intersection over Union)
- **Range**: 0 to 1
- **Interpretation**:
  - 0.50+: Good overlap with ground truth
  - 0.30-0.50: Moderate overlap
  - <0.30: Poor overlap

### Area Estimation
- Default pixel resolution: 10m (Sentinel-1 SAR)
- Formula: `Area (km²) = (oil_pixels × resolution²) / 1,000,000`
- Adjustable in web app for different satellite sources

## Customization

### Change Training Parameters

Edit `train.py`:

```python
train_model(
    num_epochs=30,        # Increase for better convergence
    batch_size=16,        # Increase if GPU memory allows
    learning_rate=1e-3    # Adjust learning rate
)
```

### Modify Model Architecture

Edit `model.py` to change:
- Number of encoder/decoder levels
- Channel dimensions
- Attention gate configuration

### Adjust Preprocessing

Edit `dataset.py` to modify:
- Image size (default: 256×256)
- Blur kernel size
- Normalization method

## Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `train.py`

**2. Dataset Not Found**
```bash
FileNotFoundError: No images found in data/train/images
```
**Solution**: Run `python organize_data.py` after downloading dataset

**3. Model Loading Error in Streamlit**
```bash
Error loading model: [Errno 2] No such file or directory: 'best_model.pth'
```
**Solution**: Train model first with `python train.py`

**4. Slow Training**
- Use GPU if available (CUDA-enabled PyTorch)
- Reduce image size or batch size
- Enable `pin_memory=True` in DataLoader

## Future Improvements

- [ ] Add data augmentation (flips, rotations, brightness)
- [ ] Implement learning rate scheduling
- [ ] Add ensemble predictions from multiple models
- [ ] Support for multiple satellite image formats
- [ ] Real-time video processing
- [ ] Integration with satellite data APIs
- [ ] Multi-class segmentation (oil types, severity)
- [ ] Export predictions to GeoJSON/Shapefile

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB

### Recommended
- GPU: NVIDIA GPU with 6+ GB VRAM (for faster training)
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 10+ GB

## License

This project is for educational and research purposes.

## Acknowledgments

- Satellite imagery: Sentinel-1 SAR (ESA Copernicus Program)
- Dataset: Kaggle Oil Spill Detection dataset
- Framework: PyTorch
- Deployment: Streamlit

## Contact

For questions or issues, please open an issue in the repository.

---

**Built with PyTorch, OpenCV, and Streamlit**

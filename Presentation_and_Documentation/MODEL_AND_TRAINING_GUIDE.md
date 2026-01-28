# U-Net Model Architecture and Training Guide

## 1. U-Net Model Architecture

The U-Net model implemented for oil spill segmentation is a lightweight version optimized for (128, 128) grayscale input images. The architecture follows the classic U-Net encoder-decoder structure with skip connections, but with reduced filter sizes to minimize computational overhead and memory footprint.

### Input Layer:
-   **Input Shape**: (128, 128, 1) - representing a 128x128 pixel grayscale image.

### Encoder Path (Contracting Path):
Each block in the encoder consists of two 3x3 convolutional layers (with ReLU activation and 'same' padding) followed by a 2x2 MaxPooling layer with strides of 2 for downsampling. The number of filters doubles at each downsampling step.

-   **Block 1:**
    -   `Conv2D(32, 3, activation='relu', padding='same')`
    -   `Conv2D(32, 3, activation='relu', padding='same')`
    -   `MaxPooling2D(pool_size=(2, 2))`
-   **Block 2:**
    -   `Conv2D(64, 3, activation='relu', padding='same')`
    -   `Conv2D(64, 3, activation='relu', padding='same')`
    -   `MaxPooling2D(pool_size=(2, 2))`
-   **Block 3:**
    -   `Conv2D(128, 3, activation='relu', padding='same')`
    -   `Conv2D(128, 3, activation='relu', padding='same')`
    -   `MaxPooling2D(pool_size=(2, 2))`

### Bottleneck Layer:
This layer connects the encoder and decoder paths and consists of two convolutional layers with increased filter depth.

-   **Bottleneck Block:**
    -   `Conv2D(256, 3, activation='relu', padding='same')`
    -   `Conv2D(256, 3, activation='relu', padding='same')`

### Decoder Path (Expansive Path):
Each block in the decoder consists of a 2x2 Conv2DTranspose layer for upsampling, a concatenation with the corresponding skip connection from the encoder, and two 3x3 convolutional layers (with ReLU activation and 'same' padding). The number of filters halves at each upsampling step.

-   **Block 1 (Upsample from Bottleneck):**
    -   `Conv2DTranspose(128, 2, strides=(2, 2), padding='same')`
    -   `Concatenate()` with output from Encoder Block 3
    -   `Conv2D(128, 3, activation='relu', padding='same')`
    -   `Conv2D(128, 3, activation='relu', padding='same')`
-   **Block 2:**
    -   `Conv2DTranspose(64, 2, strides=(2, 2), padding='same')`
    -   `Concatenate()` with output from Encoder Block 2
    -   `Conv2D(64, 3, activation='relu', padding='same')`
    -   `Conv2D(64, 3, activation='relu', padding='same')`
-   **Block 3:**
    -   `Conv2DTranspose(32, 2, strides=(2, 2), padding='same')`
    -   `Concatenate()` with output from Encoder Block 1
    -   `Conv2D(32, 3, activation='relu', padding='same')`
    -   `Conv2D(32, 3, activation='relu', padding='same')`

### Output Layer:
-   `Conv2D(1, 1, activation='sigmoid')` - A 1x1 convolutional layer with a sigmoid activation function to output a single-channel segmentation mask, where pixel values represent the probability of being an oil spill.

## 2. Data Preprocessing Pipeline

The data preprocessing pipeline ensures that the SAR images and their corresponding masks are in a suitable format for the U-Net model.

1.  **Initial Dataset Cleanup**: Misaligned mask files (those without corresponding image files) were removed from both training and validation sets to ensure a 1:1 image-to-mask correspondence.
2.  **Resizing**: All images and masks were resized to a standard dimension of (128, 128) pixels.
    -   **Images**: Resized using `PIL.Image.LANCZOS` interpolation for high-quality downsampling.
    -   **Masks**: Resized using `PIL.Image.NEAREST` interpolation to preserve binary integrity (pixel values remain 0 or 1).
3.  **Pixel Normalization**: Image pixel values were normalized to the [0, 1] range by dividing by 255.0.
4.  **SAR Filtering**: A SAR-specific speckle noise reduction filter (`skimage.restoration.denoise_wavelet` with `wavelet='db1'`, `mode='soft'`, and `sigma=0.05`) was applied to the images to mitigate speckle noise inherent in SAR data, which helps in better feature extraction.
5.  **Mask Binarization**: Masks are explicitly binarized (0 or 1) to ensure clear ground truth labels for the segmentation task.

## 3. Data Augmentation Techniques

Data augmentation was applied exclusively to the training dataset to increase its size and improve the model's generalization capabilities, making it more robust to variations in real-world data. The `albumentations` library was used for this purpose.

The augmentation pipeline includes:
-   **HorizontalFlip**: Randomly flips images horizontally with a probability of 0.5.
-   **VerticalFlip**: Randomly flips images vertically with a probability of 0.5.
-   **Rotate**: Randomly rotates images by up to +/- 30 degrees with a probability of 0.5. A `border_mode` of `cv2.BORDER_CONSTANT` and `value`/`mask_value` of 0 were used to fill empty areas created by rotation.
-   **RandomBrightnessContrast**: Randomly adjusts the brightness and contrast of images within a limit of 0.2 with a probability of 0.5.

For each original training image and mask, an augmented version was created and saved alongside the original processed data, effectively doubling the training dataset size.

## 4. Model Training Process

### Optimizer:
-   **Adam Optimizer**: Chosen for its efficiency and good performance in many deep learning tasks.
-   **Learning Rate**: Initial learning rate set to `1e-4`.

### Loss Function:
-   **BinaryCrossentropy**: Standard loss function for binary segmentation tasks, measuring the pixel-wise difference between predicted probabilities and true binary labels.

### Evaluation Metrics:
To thoroughly assess the model's performance, the following metrics were tracked during training and evaluation:
-   **MeanIoU (Mean Intersection over Union)**: Calculates the average Intersection over Union for both classes (background and spill). `num_classes` set to 2.
-   **BinaryAccuracy**: Measures the percentage of correctly classified pixels.
-   **Precision**: Quantifies the proportion of true positive predictions among all positive predictions.
-   **Recall**: Measures the proportion of true positive predictions among all actual positive samples.

### Callbacks:
Two callbacks were used to optimize the training process:
-   **EarlyStopping**: Monitors the `val_loss`. If the validation loss does not improve for `patience=10` consecutive epochs, training is stopped, and the model weights from the best performing epoch are restored.
-   **ReduceLROnPlateau**: Monitors the `val_loss`. If the validation loss does not improve for `patience=5` consecutive epochs, the learning rate is reduced by a `factor=0.5`. The learning rate will not drop below `min_lr=1e-7`.

### Mixed Precision Training:
-   **`tf.keras.mixed_precision.set_global_policy('mixed_float16')`**: Enabled to leverage the performance benefits of half-precision floating-point numbers (float16) where appropriate. This significantly speeds up training and reduces memory usage on compatible hardware (like GPUs) without a significant loss in model accuracy.

### Training Execution:
-   The model was trained for `50` epochs (though early stopping might conclude training sooner).
-   `steps_per_epoch` and `validation_steps` were calculated based on the dataset sizes and `BATCH_SIZE=32` to ensure proper iteration through the datasets.

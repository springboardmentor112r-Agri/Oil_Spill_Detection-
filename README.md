# Oil Spill Detection

## Problem Statement
Oil spills cause severe environmental damage to water bodies and agricultural ecosystems.
This project aims to detect oil spills from images using a deep learning approach.

## Approach
- Image-based binary classification
- Classes: Oil Spill / No Oil Spill
- Convolutional Neural Network (CNN)

## Dataset
### Full dataset not pushed due to size constraints 
The dataset consists of satellite images and corresponding segmentation masks.

- images/: Original input images
- masks/: Binary masks indicating oil spill regions
- train/ and val/ splits provided

Each mask highlights oil spill areas in the corresponding image.

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Project Structure
- notebooks/ : Experiments and model training
- src/ : Training and inference scripts
- data/ : Sample images
- results/ : Outputs and evaluation results

## Future Improvements
- Improve accuracy with transfer learning
- Deploy as a web application

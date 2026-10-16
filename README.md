# Brain Tumor

This project focuses on brain tumor segmentation using MRI images and a U-Net deep learning model. The implementation is contained in the notebook `Brain_Tumor_Segmentation_UNet.ipynb`, and the trained model is saved as `brain_tumor_unet.keras`.

## Overview

Brain tumor segmentation is a medical image analysis task where the model predicts the region of the tumor in MRI scans. In this project, grayscale MRI slices are processed and passed through a U-Net segmentation architecture to generate binary tumor masks.

## Project Goals

- Train a deep learning model for brain tumor segmentation
- Preprocess MRI images and corresponding masks
- Build and evaluate a U-Net model
- Save the trained model for future inference
- Visualize predictions and segmentation results

## Repository Structure

```bash
Brain_Tumor/
├── Brain_Tumor_Segmentation_UNet.ipynb   # Main notebook with training and evaluation
├── brain_tumor_unet.keras                # Trained U-Net model
├── data/
│   ├── images/                           # MRI image slices
│   └── masks/                            # Ground-truth segmentation masks
├── README.md                             # Project documentation
└── .gitignore                            # Optional Git ignore file (if present)
```

## Dataset

The project uses a dataset of MRI images and their corresponding binary masks stored under `data/images` and `data/masks`.

- Image size: 128 x 128 grayscale images
- Total samples: 500 image-mask pairs
- Train/test split: 80/20
- Mask format: binary segmentation map (0 = background, 1 = tumor)

## Model Architecture

A custom U-Net architecture is used for pixel-wise tumor segmentation.

Key components include:
- Input layer: 128 x 128 x 1 grayscale MRI
- Encoder with convolution and max-pooling blocks
- Bottleneck layer
- Decoder with upsampling and concatenation
- Final output layer with sigmoid activation for binary segmentation

## Training and Evaluation

The notebook trains the model using:
- Optimizer: Adam
- Loss: Binary Cross Entropy
- Metric: Accuracy
- Epochs: 10
- Batch size: 16

The project evaluates the model using metrics such as:
- Test Accuracy
- Dice Score
- IoU (Intersection over Union)

Sample results reported in the notebook:
- Test Accuracy: ~0.9853
- Dice Score: ~0.6727
- IoU Score: ~0.5068

## Environment Requirements

This project is built with Python and TensorFlow/Keras.

Recommended dependencies:

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```

## How to Run

1. Clone the repository
2. Open the notebook `Brain_Tumor_Segmentation_UNet.ipynb`
3. Run the notebook cells in order
4. The notebook will:
   - load MRI images and masks
   - preprocess the data
   - train the U-Net model
   - evaluate model performance
   - save the trained model to `brain_tumor_unet.keras`

## Model Inference

The trained model can be used to predict tumor masks for new MRI scans. The saved model file is:

```bash
brain_tumor_unet.keras
```

Example usage:

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("brain_tumor_unet.keras")

img = Image.open("path/to/mri_image.png").convert("L")
img_array = np.array(img, dtype="float32") / 255.0
img_array = img_array.reshape(1, 128, 128, 1)

pred = model.predict(img_array)
pred_mask = (pred > 0.5).astype("float32")
```

## Notes

- This is a research/demo-style project for learning and experimentation.
- MRI segmentation quality can vary depending on the quality and quantity of the dataset.
- The model is trained on grayscale 2D MRI slices, not 3D volumetric data.

## License

This project is for educational and research purposes. Please add an appropriate license if you plan to share or deploy it publicly.

## Acknowledgements

- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- MRI image dataset used for training and validation

## Contact

If you want to improve this project, you can extend it with:
- data augmentation
- better loss functions for imbalanced masks
- model checkpointing
- deployment as a web app or API
- 3D MRI segmentation support




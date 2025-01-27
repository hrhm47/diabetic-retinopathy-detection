
# DeepDRiD: Deep Learning for Diabetic Retinopathy Image Classification

This repository provides a deep learning-based framework for classifying diabetic retinopathy images using pre-trained models and custom training techniques. The framework leverages several ensemble learning techniques, image preprocessing methods, and model interpretability tools like GradCAM for enhanced performance and insights into model predictions.

## Overview

The project uses deep convolutional neural networks (CNNs) to predict the severity of diabetic retinopathy from retina images. The key components of the project are:

- **Data Preprocessing**: Various image preprocessing techniques to enhance model performance.
- **Model Training**: Pre-trained models (ResNet, DenseNet, EfficientNet) fine-tuned on the dataset.
- **Ensemble Learning**: Combining predictions from different models using max voting and weighted average.
- **GradCAM**: Visualizing model predictions to understand what parts of the image are being used by the model.
- **Training Logs and Visualization**: Detailed logs and plots of training and validation losses and accuracies.

- 
![ModelResult](diabetic-retinopathy-detection/src
/attention_model_results.png)

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision 0.9+
- OpenCV
- scikit-learn
- pandas
- matplotlib
- tqdm
- Pillow
- json

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https:/github.com/hrhm47/diabetic-retinopathy-detection/master
cd deepdrid
pip install -r requirements.txt
```

Make sure you have a working GPU (CUDA-enabled) if you plan to train models. The code will automatically fall back to CPU if CUDA is not available.

## Dataset

The dataset consists of retina images with labels indicating the severity of diabetic retinopathy. The dataset should be structured as follows:

```
/data/raw/DeepDRiD/train/
                        1/
                          1_l1.jpg
                          1_l2.jpg
                          ...
                    /val/
                        265/
                          265_l1.jpg
                          265_l2.jpg
                          ...
        train.csv
        val.csv
```

### CSV File Structure

Both `train.csv` and `val.csv` should contain the following columns:
- `img_path`: Relative path to the image in the dataset.
- `Overall quality`: The label indicating the severity of diabetic retinopathy (integer value from 0 to 4).

## Usage

### Training the Model

To train the model, run the following command:

```bash
python train.py
```

This will:
- Load the training and validation data from the specified directories and CSV files.
- Apply necessary transformations and preprocessing steps to the images.
- Train each model (ResNet18, DenseNet121, EfficientNet-B0) for 15 epochs.
- Log the training/validation performance in JSON format.
- Perform ensemble learning (max voting and weighted average).
- Save the ensemble results in `ensemble_results.json`.

## Code Structure

```
/data/raw
        /train           # Training images
        /val             # Validation images
        train.csv        # Training CSV file with image paths and labels
        val.csv          # Validation CSV file with image paths and labels
/src/
    train.py             # Script to train the models
/src/
    plot_training.py     # Script to plot training and validation metrics
/models                  # Contains pre-trained models and custom network definitions
```

## Model Architecture

The following models are used in this project:
- **ResNet-18**: A 18-layer residual network for image classification.
- **DenseNet-121**: A densely connected convolutional network.
- **EfficientNet-B0**: A more efficient and lightweight model with excellent performance.

### Fine-tuning the Models

The models are pre-trained on ImageNet and fine-tuned on the provided dataset. The final fully connected layers of the networks are replaced with a new layer that outputs the number of classes (5 in this case).

## Results

Ensemble learning is applied to the models, combining their predictions using two techniques:
- **Max Voting**: The class with the highest count across the models' predictions is chosen.
- **Weighted Average**: A weighted average of the models' predictions is calculated, and the class with the highest value is selected.

Cohen’s Kappa score is used to evaluate the performance of the ensemble predictions.


## Acknowledgments

- PyTorch and torchvision for deep learning model implementations.
- OpenCV and PIL for image processing.
- scikit-learn for machine learning utilities like Kappa score.
- GradCAM for model interpretability.


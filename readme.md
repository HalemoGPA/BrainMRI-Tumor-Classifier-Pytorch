
# Brain Tumor Classification

This repository contains a deep learning-based solution for classifying brain tumors using MRI images. The model is trained to classify images into five categories: No Tumor, Pituitary, Glioma, Meningioma, and Other.

## Table of Contents
- [Brain Tumor Classification](#brain-tumor-classification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Requirements](#requirements)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Streamlit App](#streamlit-app)
  - [Usage](#usage)
  - [Results](#results)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)
  - [Next Steps:](#next-steps)

## Overview
This project uses a Convolutional Neural Network (CNN) implemented in PyTorch to classify brain MRI images. The model architecture consists of multiple convolutional, batch normalization, max-pooling layers followed by fully connected layers.

## Dataset
The dataset used is the Brain Tumor MRI Dataset available on Kaggle. It contains MRI images for training and testing the model.

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

Install the required packages using:
```sh
pip install -r requirements.txt
```

## Training
The training script preprocesses the images, defines the model architecture, and trains the model.

1. **Preprocessing:** Images are resized and normalized.
2. **Model Architecture:** Defined in `model.py`.
3. **Training Loop:** Defined in the notebook with performance metrics.

Run the training script:
```sh
python train.py
```

## Evaluation
The trained model is evaluated on a validation set, and the best-performing model is saved. The evaluation metrics include accuracy and loss.

Run the evaluation script:
```sh
python evaluate.py
```

## Streamlit App
A Streamlit application is provided for deploying the model and making predictions on new MRI images.

1. **Load the model**: The pre-trained model is loaded.
2. **Upload Image**: Upload an MRI image to the app.
3. **Prediction**: The app displays the predicted tumor type.

Run the Streamlit app:
```sh
streamlit run app.py
```

## Usage
1. **Clone the repository**:
```sh
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```
2. **Install dependencies**:
```sh
pip install -r requirements.txt
```
3. **Train the model**:
```sh
python train.py
```
4. **Evaluate the model**:
```sh
python evaluate.py
```
5. **Run the Streamlit app**:
```sh
streamlit run app.py
```

## Results
The model's performance is visualized through training and validation loss and accuracy plots. Confusion matrices are also used to illustrate the classification performance on the test set.

## Acknowledgments
- The dataset is provided by [Navoneel Chakrabarty](https://www.kaggle.com/navoneel) on Kaggle.
- This project uses PyTorch for building and training the model.

## License
This project is licensed under the MIT License.


## Next Steps:
1. **Project Organization**: 
    - Organize your repository with the following structure:
      ```
      brain-tumor-classification/
      ├── data/
      ├── models/
      ├── notebooks/
      ├── src/
      │   ├── __init__.py
      │   ├── dataset.py
      │   ├── model.py
      │   ├── train.py
      │   ├── evaluate.py
      │   ├── utils.py
      ├── app.py
      ├── requirements.txt
      ├── README.md
      └── .gitignore
      ```

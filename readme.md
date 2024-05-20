
# Brain Tumor Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a deep learning-based solution for classifying brain tumors using MRI images. The model is trained to classify images into four categories: No Tumor, Pituitary, Glioma, Meningioma

## Table of Contents
- [Brain Tumor Classification](#brain-tumor-classification)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Requirements](#requirements)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Streamlit App](#streamlit-app)
    - [Functionality:](#functionality)
  - [Usage](#usage)
  - [Results](#results)
  - [Acknowledgments](#acknowledgments)

## Overview
This project uses a Convolutional Neural Network (CNN) implemented in PyTorch to classify brain MRI images. The model architecture consists of multiple convolutional, batch normalization, max-pooling layers followed by fully connected layers.

## Dataset
The dataset used is the Brain Tumor MRI Dataset available on Kaggle. It contains MRI images for training and testing the model.

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

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



## Evaluation
The trained model is evaluated on a validation set, and the best-performing model is saved. The evaluation metrics include accuracy and loss.



## Streamlit App

A Streamlit application has been developed to facilitate the deployment of the model and enable predictions on new MRI images. The app can be accessed [here](https://st.brainmri.com).

### Functionality:

1. **Model Loading**: The pre-trained model is loaded automatically upon accessing the app.
2. **Image Upload**: Users can upload MRI images directly to the app interface.
3. **Prediction Display**: Once an image is uploaded, the app displays the predicted tumor type based on the model's classification.

The Streamlit app provides a user-friendly interface for interacting with the model and obtaining predictions effortlessly.

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

3. **Run the Streamlit app**:
```sh
streamlit run app.py
```


## Results
The model achieves an accuracy of 99.3% on the test set. Training and validation loss and accuracy plots are provided to visualize the model's performance. Confusion matrices illustrate the classification performance on the test set.

## Acknowledgments
- The dataset is provided by [MASOUD NICKPARVAR](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle.
- This project uses PyTorch for building and training the model.



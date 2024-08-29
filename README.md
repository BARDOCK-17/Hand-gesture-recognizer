# Hand Gesture Recognizer

This repository contains the code for a Hand Gesture Recognizer built using the Sign Language MNIST dataset. The project uses Convolutional Neural Networks (CNNs) to recognize sign language gestures and deploys the model in a user-friendly Streamlit application.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to create an efficient model to recognize American Sign Language (ASL) hand gestures from images. The model is trained on the Sign Language MNIST dataset and is deployed using Streamlit, enabling users to interact with the model through a web application.

## Dataset

The [Sign Language MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist) consists of 28x28 grayscale images of hand gestures representing the letters A-Z (except J and Z, which require motion).

- **Classes**: 24 (A-Y excluding J, Z)
- **Number of images**: 27,455 training images, 7,172 test images

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognizer.git
   cd hand-gesture-recognizer
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model or run the Streamlit app, follow the instructions below:

- **Train the model:**
   ```bash
   python train_model.py
   ```

- **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Model Architecture

The model is built using CNNs to capture the spatial features of the hand gestures. It includes several convolutional layers followed by max-pooling and dropout layers to prevent overfitting.

- **Input Layer**: 28x28 grayscale images
- **Convolutional Layers**: Extract features from the input images
- **Max Pooling Layers**: Downsample the feature maps
- **Dropout Layers**: Regularization to prevent overfitting
- **Fully Connected Layers**: Final classification

## Results

The model achieves an accuracy of **XX%** on the test set. (You can add specific results here once your model is trained.)

## Streamlit App

The Streamlit app provides a simple interface for users to upload an image of a hand gesture and get a prediction for the corresponding letter.

- **Features**:
  - Upload an image for prediction
  - View the model's prediction
  - Visualize the confidence levels for each class

## Future Work

- **Enhancing Model Accuracy**: Experiment with different model architectures and hyperparameter tuning.
- **Expand Dataset**: Include additional sign language datasets that cover gestures for letters J and Z.
- **Improve UI**: Add more features to the Streamlit app for better user experience.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

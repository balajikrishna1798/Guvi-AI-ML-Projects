# User Guide: Plant Disease Detection App

## Overview
This application helps you identify plant diseases by simply uploading a picture of a plant leaf. The system uses a trained Machine Learning model to analyze the image and predict the disease.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`)

### 1. Training the Model
Before running the application for the first time, you need a trained model.
1. Download the "New Plant Diseases Dataset" from Kaggle.
2. Extract the dataset into a folder named `dataset/` in the project root. The folder structure should be:
   ```
   Project2_Plant_Disease_Detection/
   ├── dataset/
       ├── train/
           ├── Class1/
           ├── Class2/
       ├── valid/
           ├── Class1/
           ├── Class2/
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
   *This script will train the Custom CNN along with ResNet18, VGG16, and MobileNetV2, and save the best model weights.*

### 2. Running the Application
Once you have the `CustomCNN_best.pth` file in your directory:
1. Open your terminal.
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Your browser will open automatically.

## How to Use the App
1. Click the **"Browse files"** button or drag and drop an image of a leaf.
2. Ensure the image is clear and shows the leaf prominently.
3. The app will automatically process the image and display the predicted disease and a confidence score.
4. You can provide feedback on the prediction using the Yes/No buttons.

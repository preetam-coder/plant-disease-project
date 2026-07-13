# 🌿 Plant Disease Detection

A smart plant leaf disease detection project built with Python, TensorFlow, and Streamlit. The app lets users upload an image of a plant leaf, predicts whether it is healthy or diseased, displays the confidence score, and provides AI-generated treatment suggestions for detected diseases.

## Project Overview

This project uses a trained Convolutional Neural Network (CNN) model to classify plant diseases from leaf images. It is designed to help farmers, students, and agricultural workers quickly identify common plant issues and take appropriate action.

## Features

- Upload leaf images in JPG, JPEG, or PNG format
- CNN-based image classification for plant diseases
- Healthy leaf detection
- Confidence score for each prediction
- AI-powered treatment recommendations using Mistral
- PDF report download for the detected disease and recommendation
- Modern Streamlit web interface

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Pillow
- NumPy
- ReportLab
- Python-dotenv
- Mistral AI API

## Project Workflow

1. The user uploads an image of a plant leaf.
2. The app preprocesses the image by resizing it to 224 × 224 and normalizing the pixel values.
3. The trained CNN model predicts the disease class.
4. The predicted class is mapped to the human-readable label from the class index file.
5. If the leaf is not healthy, the app generates a treatment recommendation using Mistral.
6. The user can download the final diagnosis and treatment advice as a PDF report.

## Model and Data

The trained model file is included in the project and is loaded by the app at runtime. The model predicts among multiple disease categories and healthy classes that are stored in the label mapping file.

## Model Accuracy

Based on the training notebook output, the CNN model achieved the following results:

- Final training accuracy: 98.04%
- Best validation accuracy observed during training: 87.52%
- Final validation accuracy recorded by the notebook evaluation: 87.53%

These values indicate that the model performs well on the plant disease dataset and is able to classify leaf images with strong recognition capability.

## Result / Output

After uploading an image, the app shows:

- The detected disease name or healthy status
- Prediction confidence in percentage
- Condition-specific treatment guidance
- A downloadable PDF report for documentation

Example result labels include classes such as:

- Apple scab
- Black rot
- Cedar apple rust
- Tomato late blight
- Potato healthy
- Grape leaf blight

## Installation

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your Mistral API key in a `.env` file:

```bash
MISTRAL_API_KEY=your_api_key_here
```

## Run the Application

```bash
streamlit run main.py
```

## Project Files

- [main.py](main.py) — Streamlit app and prediction pipeline
- [final_plant_disease_prediction_model.h5](final_plant_disease_prediction_model.h5) — trained CNN model
- [class_indices.json](class_indices.json) — class label mapping
- [requirements.txt](requirements.txt) — Python dependencies
- [final_Plant_Disease_Prediction_CNN_Image_Classifier_(1).ipynb](final_Plant_Disease_Prediction_CNN_Image_Classifier_(1).ipynb) — training notebook

## Summary

This project provides a practical solution for plant disease recognition using deep learning and an interactive UI. It is useful for quick disease screening and can be extended with more plant types, larger datasets, and improved model accuracy in future versions.

The model currently shows a validation accuracy of 87.53% based on the notebook evaluation, with the app delivering disease classification, confidence score, treatment guidance, and downloadable PDF reporting.

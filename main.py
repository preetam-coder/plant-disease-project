import os
import json
import gdown
import re
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

try:
    from groq import Groq
except ImportError:
    Groq = None
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    working_dir,
    "final_plant_disease_prediction_model.h5"
)

class_indices_path = os.path.join(
    working_dir,
    "class_indices.json"
)

# Download model automatically if it doesn't exist
if not os.path.exists(model_path):
    with st.spinner("Downloading AI model... This may take a few minutes on the first run."):
        gdown.download(
            id="1p8aNPIi4ihoqzWe8KPSHQ1gDrL65soVv",
            output=model_path,
            quiet=False
        )

# Load environment variables from the project root
load_dotenv(os.path.join(working_dir, ".env"))

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_indices():
    with open(class_indices_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load model
model = load_trained_model()

# Load class indices
class_indices = load_class_indices()

# Configure Groq AI
api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

client = None
if api_key and Groq is not None:
    client = Groq(api_key=api_key)

# ---------- Functions ----------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indices[str(predicted_class_index)]
    confidence = float(predictions[0][predicted_class_index])
    return predicted_class, confidence

def is_healthy(prediction):
    """Check if the prediction indicates a healthy leaf"""
    return "healthy" in prediction.lower()

def get_disease_solution(disease_name):
    """Generate treatment recommendations using Groq AI."""
    try:
        if not api_key:
            return "⚠️ GROQ_API_KEY not configured. Add it to your environment variables."

        if Groq is None:
            return "⚠️ Groq SDK is unavailable. Install it with: pip install groq"

        if client is None:
            return "⚠️ Groq client could not be initialized. Check your GROQ_API_KEY."

        prompt = f"""
You are an agricultural expert.

Provide a concise, practical treatment solution for the plant disease: {disease_name}

Please structure your response as follows:

1. Disease Overview
   - Brief description (2-3 sentences)

2. Treatment Methods
   - 3-5 actionable treatments

3. Prevention Tips
   - 2-3 prevention strategies

Keep the response clear, practical, and under 300 words.
If the disease identification is uncertain, advise the user to confirm it with a local agricultural expert.
"""

        response = client.chat.completions.create(
            model=groq_model,

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
    from mistralai.client import Mistral
except Exception:
    try:
        from mistralai import Mistral
    except Exception:
        Mistral = None
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
            id="id_value of trained model",
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

# Configure Mistral AI
api_key = os.getenv("MISTRAL_API_KEY")

client = None
if api_key and Mistral is not None:
    client = Mistral(api_key=api_key)

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
    """Generate treatment solution using Mistral AI"""

    try:
        if not api_key:
            return "⚠️ MISTRAL_API_KEY not configured."

        if client is None:
            return "⚠️ Mistral client is unavailable in this environment."

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
"""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error generating treatment recommendation: {str(e)}"

def generate_pdf_report(disease_name, solution_text):
    """Generate a PDF report for the disease and treatment solution"""
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0f2027'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Disease name style
    disease_style = ParagraphStyle(
        'DiseaseStyle',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#ff6b6b'),
        spaceAfter=20,
        spaceBefore=10,
        fontName='Helvetica-Bold',
        borderPadding=10,
        borderColor=colors.HexColor('#ff6b6b'),
        borderWidth=2,
        alignment=TA_CENTER
    )
    
    # Solution style
    solution_style = ParagraphStyle(
        'SolutionStyle',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#2c5364'),
        spaceAfter=12,
        leading=16,
        alignment=TA_LEFT
    )
    
    # Metadata style
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    
    # Add title
    title = Paragraph("🌿 Plant Disease Detection Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    date_para = Paragraph(f"Generated on: {timestamp}", meta_style)
    elements.append(date_para)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add disease name heading
    disease_display = disease_name.replace('___', ' - ').replace('_', ' ')
    disease_heading = Paragraph(f"<b>Detected Disease:</b><br/>{disease_display}", disease_style)
    elements.append(disease_heading)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add section heading for treatment
    treatment_heading = Paragraph("<b>Treatment Recommendations:</b>", 
                                 ParagraphStyle('TreatmentHeading',
                                              parent=styles['Heading3'],
                                              fontSize=14,
                                              textColor=colors.HexColor('#00cc99'),
                                              spaceAfter=15,
                                              fontName='Helvetica-Bold'))
    elements.append(treatment_heading)
    
    # Add solution text (convert markdown-like formatting to HTML properly)
    # First, escape any existing HTML characters
    solution_html = solution_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Convert markdown bold (**text**) to HTML <b>text</b> using regex
    solution_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', solution_html)
    
    # Convert ### headings to bold
    solution_html = re.sub(r'###\s*(.+?)(?=\n|$)', r'<b>\1</b>', solution_html)
    
    # Convert newlines to <br/> tags
    solution_html = solution_html.replace('\n\n', '<br/><br/>')
    solution_html = solution_html.replace('\n', '<br/>')
    
    # Remove any remaining asterisks or problematic markdown syntax
    solution_html = solution_html.replace('*', '•')  # Convert bullet asterisks to bullet points
    
    solution_para = Paragraph(solution_html, solution_style)
    elements.append(solution_para)
    
    # Add footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = Paragraph(
        "<i>This report is generated by an AI-powered Plant Disease Detection System. "
        "For professional agricultural advice, please consult with a certified plant pathologist.</i>",
        meta_style
    )
    elements.append(footer_text)
    
    # Build PDF
    doc.build(elements)
    
    buffer.seek(0)
    return buffer

# ---------- Custom CSS ----------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Background image with overlay */
        .stApp {
            background: linear-gradient(rgba(15, 32, 39, 0.85), rgba(44, 83, 100, 0.85));
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
            font-family: 'Poppins', sans-serif;
        }

        /* Title styling */
        h1 {
            text-align: center;
            background: linear-gradient(90deg, #00ff99, #00ccff, #00ff99);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 3s ease infinite;
            font-weight: 700;
            text-shadow: 0 0 30px rgba(0, 255, 153, 0.3);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #00ff99, #00ccff);
            color: #0f2027;
            border-radius: 15px;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 255, 153, 0.4);
            transition: all 0.3s ease-in-out;
            width: 100%;
        }
        .stButton>button:hover {
            transform: scale(1.05) translateY(-2px);
            background: linear-gradient(45deg, #00ccff, #00ff99);
            box-shadow: 0 6px 20px rgba(0, 204, 255, 0.6);
        }
        .stButton>button:active {
            transform: scale(0.98);
        }

        /* File uploader */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 2px dashed rgba(0, 255, 153, 0.3);
            transition: all 0.3s ease;
        }
        .stFileUploader:hover {
            border-color: rgba(0, 255, 153, 0.6);
            background: rgba(255, 255, 255, 0.08);
        }

        /* Image container */
        .uploaded-img {
            transition: transform 0.4s ease;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        .uploaded-img:hover {
            transform: scale(1.03) rotate(1deg);
        }

        /* Prediction box - Healthy */
        .prediction-box-healthy {
            padding: 20px;
            background: linear-gradient(135deg, rgba(0, 255, 153, 0.2), rgba(0, 204, 255, 0.2));
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: 700;
            animation: fadeIn 0.8s ease-in-out;
            border: 2px solid rgba(0, 255, 153, 0.5);
            box-shadow: 0 4px 20px rgba(0, 255, 153, 0.3);
            margin: 15px 0;
        }

        /* Prediction box - Disease */
        .prediction-box-disease {
            padding: 20px;
            background: linear-gradient(135deg, rgba(255, 100, 100, 0.2), rgba(255, 150, 50, 0.2));
            border-radius: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            animation: fadeIn 0.8s ease-in-out;
            border: 2px solid rgba(255, 100, 100, 0.5);
            box-shadow: 0 4px 20px rgba(255, 100, 100, 0.3);
            margin: 15px 0;
        }

        /* Confidence score */
        .confidence-score {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            font-size: 14px;
            margin-top: 10px;
            font-weight: 500;
        }

        /* Solution box */
        .solution-box {
            padding: 25px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            margin-top: 20px;
            border-left: 4px solid #00ff99;
            animation: slideIn 0.6s ease-in-out;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            line-height: 1.8;
        }
        
        .solution-box h3 {
            color: #00ff99;
            margin-bottom: 15px;
        }

        /* Animations */
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(20px);}
            to {opacity: 1; transform: translateY(0);}
        }
        
        @keyframes slideIn {
            from {opacity: 0; transform: translateX(-20px);}
            to {opacity: 1; transform: translateX(0);}
        }
        
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Subheader styling */
        .stSubheader {
            color: #00ff99 !important;
            font-weight: 600;
        }
        
        /* Info boxes */
        .stAlert {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border-left: 4px solid #00ccff;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- App Title ----------
st.markdown("<h1>🌿 Plant Disease Detection 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px; color:#00ccff; margin-bottom:30px;'>Upload a leaf image to detect diseases and get AI-powered treatment solutions</p>", unsafe_allow_html=True)

# Upload image
st.subheader("📤 Upload an image of a leaf")
uploaded_image = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((250, 250))
        st.image(resized_img, caption="Uploaded Image", use_container_width=False)

    with col2:
        if st.button('🔍 Classify Disease'):
            with st.spinner('🔬 Analyzing leaf...'):
                prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                
                # Store prediction in session state
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                
                # Check if healthy
                if is_healthy(prediction):
                    st.markdown(
                        f"""<div class='prediction-box-healthy'>
                            <div style='font-size:28px; margin-bottom:10px;'>✨ The leaf is healthy! ✨</div>
                            <div style='font-size:16px; color:#e0e0e0;'>No disease detected</div>
                            <div class='confidence-score'>Confidence: {confidence*100:.2f}%</div>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                else:
                    # Display disease prediction
                    disease_display = prediction.replace('___', ' - ').replace('_', ' ')
                    st.markdown(
                        f"""<div class='prediction-box-disease'>
                            <div style='font-size:18px; margin-bottom:5px;'>⚠️ Disease Detected</div>
                            <div style='font-size:24px; color:#ff6b6b; font-weight:700;'>{disease_display}</div>
                            <div class='confidence-score'>Confidence: {confidence*100:.2f}%</div>
                        </div>""", 
                        unsafe_allow_html=True
                    )

    # Show solution button only if disease is detected
    if hasattr(st.session_state, 'prediction') and not is_healthy(st.session_state.prediction):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button('💊 Get Treatment Solution'):
                with st.spinner('🤖 AI is generating treatment recommendations...'):
                    solution = get_disease_solution(st.session_state.prediction)
                    st.session_state.solution = solution
        
        # Display solution if available
        if hasattr(st.session_state, 'solution'):
            st.markdown(
                f"""<div class='solution-box'>
                    <h3>🌱 Treatment Recommendations</h3>
                    {st.session_state.solution}
                </div>""", 
                unsafe_allow_html=True
            )
            
            # Add PDF download button
            st.markdown("<br/>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Generate PDF
                pdf_buffer = generate_pdf_report(st.session_state.prediction, st.session_state.solution)
                
                # Create filename with disease name and timestamp
                disease_safe_name = st.session_state.prediction.replace('___', '_').replace(' ', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"PlantDisease_Report_{disease_safe_name}_{timestamp}.pdf"
                
                # Download button
                st.download_button(
                    label="📥 Download Report as PDF",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    help="Download a professional PDF report with disease information and treatment recommendations"
                )
else:
    st.info("👆 Please upload a leaf image to get started")

# Footer
st.markdown("<hr style='margin-top:50px; border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:rgba(255,255,255,0.5); font-size:14px;'>Powered by TensorFlow & Google AI • Plant Disease Detection System</p>", unsafe_allow_html=True)

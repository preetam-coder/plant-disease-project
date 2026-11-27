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
st.markdown("<p style='text-align:center; color:rgba(255,255,255,0.5); font-size:14px;'>Powered by TensorFlow & Groq AI • Plant Disease Detection System</p>", unsafe_allow_html=True)

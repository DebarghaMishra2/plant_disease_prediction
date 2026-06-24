import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="Plant Disease Detector", layout="centered")

# Load the model (ensure the .keras file is in the same directory)
@st.cache_resource
def load_plant_model():
    model = tf.keras.models.load_model('Plant_Disease_Model.keras')
    return model

# Hardcoded class names (as derived from your dataset structure)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to identify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            model = load_plant_model()
            
            # Preprocessing (matches our successful Colab tests)
            img = np.array(image.convert('RGB'))
            img_resized = cv2.resize(img, (128, 128))
            img_array = np.expand_dims(np.array(img_resized, dtype='float32'), axis=0)
            
            # Prediction
            predictions = model.predict(img_array)
            result_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            st.success(f"Prediction: **{class_names[result_idx]}**")
            st.info(f"Confidence: **{confidence:.2f}%**")

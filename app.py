import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

# Fix compatibility
tf.keras.utils.get_custom_objects().clear()

# Download model
url = "https://drive.google.com/uc?id=1zPXNUbJrXxFSUXuWZ2PxIR7e5EinuGD_"
output = "plant_disease_model.h5"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load model
model = load_model(output, compile=False, safe_mode=False)

# Labels
class_labels = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___healthy',
    'Corn___Common_rust','Corn___healthy','Potato___Early_blight',
    'Potato___Late_blight','Tomato___Bacterial_spot',
    'Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites','Tomato___Target_Spot',
    'Tomato___healthy'
]

# Treatments
treatments = {
    'Apple___Apple_scab': "Apply fungicide and remove infected leaves",
    'Apple___Black_rot': "Prune affected areas and use fungicide",
    'Apple___Apple_scab': "Apply fungicide and remove infected leaves",
    'Apple___healthy': "No treatment needed",
    'Corn___Common_rust': "Use resistant varieties and fungicides",
    'Corn___healthy': "Healthy plant",
    'Potato___Early_blight': "Use copper-based fungicides",
    'Potato___Late_blight': "Remove infected plants and apply fungicide",
    'Tomato___Bacterial_spot': "Use copper sprays",
    'Tomato___Early_blight': "Apply fungicide",
    'Tomato___Late_blight': "Remove infected plants",
    'Tomato___Leaf_Mold': "Improve air circulation",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves",
    'Tomato___Spider_mites': "Use neem oil",
    'Tomato___Target_Spot': "Apply fungicide",
    'Tomato___healthy': "Healthy plant"
}

# UI
st.title("🌿 Plant Disease Detection")

file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = np.argmax(pred)
    confidence = np.max(pred)*100

    disease = class_labels[index]

    st.success(f"Prediction: {disease}")
    st.info(f"Confidence: {confidence:.2f}%")

    if disease in treatments:
        st.warning(treatments[disease])

import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from keras.models import load_model

# 🔽 Download model from Google Drive
url = "https://drive.google.com/uc?id=1zPXNUbJrXxFSUXuWZ2PxIR7e5EinuGD_"
output = "model.h5"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# 🔽 Load model
model = load_model(output, compile=False, safe_mode=False)

# 🔽 Class labels
class_labels = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___healthy',
    'Corn___Common_rust','Corn___healthy','Potato___Early_blight',
    'Potato___Late_blight','Tomato___Bacterial_spot',
    'Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites','Tomato___Target_Spot',
    'Tomato___healthy'
]

# 🔽 Treatment suggestions
treatments = {
    'Apple___Apple_scab': "Apply fungicide and remove infected leaves",
    'Apple___Black_rot': "Prune affected areas and use fungicide",
    'Apple___healthy': "No treatment needed",
    
    'Corn___Common_rust': "Use resistant varieties and fungicides",
    'Corn___healthy': "Healthy plant",
    
    'Potato___Early_blight': "Use copper-based fungicides",
    'Potato___Late_blight': "Remove infected plants and apply fungicide",
    
    'Tomato___Bacterial_spot': "Use copper sprays and remove infected leaves",
    'Tomato___Early_blight': "Apply fungicide regularly",
    'Tomato___Late_blight': "Remove infected plants immediately",
    'Tomato___Leaf_Mold': "Improve air circulation and use fungicide",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves and apply fungicide",
    'Tomato___Spider_mites': "Use insecticidal soap or neem oil",
    'Tomato___Target_Spot': "Apply fungicide and avoid overhead watering",
    'Tomato___healthy': "Healthy plant"
}

# 🔽 Streamlit UI
st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    disease = class_labels[predicted_index]

    st.success(f"Prediction: {disease}")
    st.info(f"Confidence: {confidence:.2f}%")

    if disease in treatments:
        st.warning(f"Treatment: {treatments[disease]}")

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests

# Download model from Google Drive
@st.cache_resource
def load_model():
    url = "PASTE_YOUR_DRIVE_LINK_HERE"
    model_path = "model.h5"
    
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = [
    "Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"
]

st.title("🌿 Plant Disease Detection (Real Model)")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction)*100, 2)

    st.success(f"🌱 Disease: {result}")
    st.info(f"📊 Confidence: {confidence}%")

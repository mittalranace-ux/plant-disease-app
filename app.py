import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("plant_disease_resnet50.h5")

# Class names (auto load)
import os
classes = os.listdir("dataset")

st.title("🌿 Plant Disease Detection App")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")

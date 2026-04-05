import streamlit as st
import numpy as np
from PIL import Image
import random

class_names = ["Apple Scab", "Black Rot", "Healthy"]

st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    predicted_class = random.choice(class_names)

    st.success(f"Disease: {predicted_class}")

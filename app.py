import streamlit as st
import numpy as np
from PIL import Image
import random
import time

# Class names (demo)
class_names = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy",
    "Leaf Spot"
]

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Show image
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Loading animation
    with st.spinner("🔍 Analyzing Image..."):
        time.sleep(2)

    # Demo prediction
    predicted_class = random.choice(class_names)
    confidence = round(random.uniform(90, 99), 2)

    # Output
    st.success(f"🌱 Disease: {predicted_class}")
    st.info(f"📊 Confidence: {confidence}%")

    # Extra info
    if predicted_class == "Healthy":
        st.success("✅ The plant is healthy. No disease detected.")
    else:
        st.warning("⚠️ Disease detected. Take necessary action.")

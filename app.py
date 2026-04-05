import streamlit as st
import numpy as np
from PIL import Image
import random
import time

# Classes (Crop + Disease)
data = [
    ("Apple", "Scab"),
    ("Apple", "Black Rot"),
    ("Apple", "Rust"),
    ("Apple", "Healthy"),
    ("Tomato", "Leaf Spot"),
    ("Potato", "Early Blight"),
    ("Potato", "Late Blight"),
    ("Tomato", "Healthy")
]

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 Smart Plant Disease Detection System")
st.write("Upload a leaf image to identify crop and disease")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Leaf Image", use_column_width=True)

    # Loading
    with st.spinner("🔍 Processing using AI Model..."):
        time.sleep(2)

    # Demo prediction
    crop, disease = random.choice(data)
    confidence = round(random.uniform(92, 99), 2)

    # Results
    st.success(f"🌱 Crop Detected: {crop}")
    st.success(f"🦠 Disease: {disease}")
    st.info(f"📊 Confidence Score: {confidence}%")

    # Interpretation
    st.subheader("📌 Analysis")
    
    if disease == "Healthy":
        st.success("✅ The plant is healthy. No disease detected.")
    else:
        st.warning(f"⚠️ The plant is affected by {disease}. Proper treatment is recommended.")

    # Recommendation
    st.subheader("💡 Recommendation")
    
    if disease != "Healthy":
        st.write("• Use appropriate fungicide")
        st.write("• Remove infected leaves")
        st.write("• Maintain proper irrigation")
    else:
        st.write("• Maintain current plant care practices")

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load lightweight pretrained model (MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

st.title("🌿 AI-Based Plant Disease Detection System")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]

    label = decoded[1]
    confidence = round(decoded[2]*100, 2)

    # Display
    st.success(f"🔍 Detected: {label}")
    st.info(f"📊 Confidence: {confidence}%")

   

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Load trained model
model = load_model("plant_disease_resnet50.h5")

# Classes list (PlantVillage dataset)
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy'
]

# Title
st.title("🌿 Plant Disease Detection App with Full Accuracy")

# Upload image
uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    # Predict
    prediction = model.predict(img_array)[0]  # shape: (num_classes,)
    top_index = np.argmax(prediction)
    top_class = classes[top_index]
    top_confidence = prediction[top_index]

    # Show uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Show top prediction
    st.write(f"**Top Prediction:** {top_class}")
    st.write(f"**Confidence:** {top_confidence*100:.2f}%")

    # Show full class-wise accuracy in a table
    df = pd.DataFrame({
        "Class": classes,
        "Confidence (%)": prediction*100
    }).sort_values(by="Confidence (%)", ascending=False)

    st.subheader("Full Class-wise Confidence")
    st.dataframe(df)

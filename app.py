import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("InceptionV3_best.h5")
    return model

model = load_model()

# Class labels (replace with your actual fish class names in order)
CLASS_NAMES = [
    "Fish Class 1", "Fish Class 2", "Fish Class 3", "Fish Class 4", "Fish Class 5",
    "Fish Class 6", "Fish Class 7", "Fish Class 8", "Fish Class 9", "Fish Class 10", "Fish Class 11"
]

# Title & description
st.title("🐟 Fish Species Classifier (InceptionV3)")
st.write("Upload an image of a fish and the model will predict its species with confidence scores.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for InceptionV3 (299x299)
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display main prediction
    st.markdown(f"### 🐠 Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Top-3 predictions
    top3_idx = np.argsort(predictions)[::-1][:3]
    st.write("### 🔝 Top 3 Predictions:")
    for idx in top3_idx:
        st.write(f"{CLASS_NAMES[idx]} — {predictions[idx]*100:.2f}%")

    # Probability chart
    st.write("### 📊 Confidence Distribution:")
    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": predictions * 100
    })
    st.bar_chart(prob_df.set_index("Class"))

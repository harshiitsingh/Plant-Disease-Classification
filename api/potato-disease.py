import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

MODEL = tf.keras.models.load_model("api/potato-disease.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

st.title("Potato Disease Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Resize image to (224, 224)
    image = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.subheader("Prediction Result:")
    st.write("Class:", predicted_class)
    st.write("Confidence:", confidence)

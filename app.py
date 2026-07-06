import streamlit as st
import numpy as np
from PIL import Image
import joblib
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("cat_dog_model.pkl")

model = load_model()

IMG_SIZE = 64

# -----------------------------
# Title
# -----------------------------
st.title("🐱 Cat vs 🐶 Dog Classifier")
st.write("Upload an image and let the AI predict whether it is a **Cat** or a **Dog**.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        # Read image using PIL
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert to numpy array
        image_np = np.array(image)

        # Resize
        image_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))

        # Flatten
        image_flatten = image_resized.flatten()

        # Prediction
        prediction = model.predict([image_flatten])[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([image_flatten])[0]
        else:
            probability = None

        # Display Result
        if prediction == 0:
            st.success("🐱 Prediction: CAT")
        else:
            st.success("🐶 Prediction: DOG")

        # Display Confidence
        if probability is not None:
            st.write(f"**🐱 Cat Probability:** {probability[0]*100:.2f}%")
            st.write(f"**🐶 Dog Probability:** {probability[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")

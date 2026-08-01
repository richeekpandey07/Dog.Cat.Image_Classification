import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import cv2

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="🐱 Cat vs Dog AI Classifier",
    page_icon="🐶",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    return joblib.load("cat_dog_model.pkl")

model = load_model()

IMG_SIZE = 64

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.feature-card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #333;
    margin-bottom: 10px;
}

.dev-card {
    background: linear-gradient(135deg,#141E30,#243B55);
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/616/616408.png",
    width=120
)

st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
### Richeek Pandey

🎓 B.Tech IT Student

🤖 AI / ML Enthusiast

📊 Data Science Learner
""")

st.sidebar.markdown(
    "[🔗 LinkedIn](https://www.linkedin.com/in/richeek-pandey-9954783a9)"
)

st.sidebar.markdown(
    "[💻 GitHub](https://github.com/richeekpandey07)"
)

st.sidebar.markdown("---")

st.sidebar.subheader("📋 Project Details")

st.sidebar.info("""
**Project:** Cat vs Dog Classifier

**Model:** Scikit-Learn

**Framework:** Streamlit

**Classes:** Cat & Dog

**Input Size:** 64 x 64

**Version:** 1.0
""")

# =====================================
# HEADER
# =====================================
st.markdown("""
<div style='text-align:center'>
<h1>🐱 VS 🐶</h1>
<h3>AI Powered Pet Recognition System</h3>
<p>Machine Learning • Computer Vision • Streamlit</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================
# FEATURES
# =====================================
st.subheader("✨ Smart Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("📸 Upload JPG / PNG Images")
    st.success("⚡ Instant Prediction")

with col2:
    st.success("🤖 Machine Learning Powered")
    st.success("📊 Confidence Scores")

with col3:
    st.success("🎨 Interactive Dashboard")
    st.success("🌐 Deployment Ready")

st.markdown("---")

# =====================================
# IMAGE UPLOAD
# =====================================
st.subheader("📤 Upload Your Pet Image")

uploaded_file = st.file_uploader(
    "Choose a Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================
if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        image_np = np.array(image)

        image_resized = cv2.resize(
            image_np,
            (IMG_SIZE, IMG_SIZE)
        )

        image_flatten = image_resized.flatten()

        prediction = model.predict([image_flatten])[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(
                [image_flatten]
            )[0]
        else:
            probability = None

        with col2:

            st.subheader("🤖 Prediction Result")

            if prediction == 0:
                st.success("🐱 CAT DETECTED")
            else:
                st.success("🐶 DOG DETECTED")

            if probability is not None:

                cat_prob = float(probability[0])
                dog_prob = float(probability[1])

                st.markdown("### 📊 Confidence Score")

                st.write(
                    f"🐱 Cat : {cat_prob*100:.2f}%"
                )
                st.progress(cat_prob)

                st.write(
                    f"🐶 Dog : {dog_prob*100:.2f}%"
                )
                st.progress(dog_prob)

        # =====================================
        # CHART
        # =====================================
        if probability is not None:

            st.markdown("---")
            st.subheader("📈 Prediction Analytics")

            chart_data = pd.DataFrame({
                "Class": ["Cat", "Dog"],
                "Probability": [
                    cat_prob * 100,
                    dog_prob * 100
                ]
            })

            st.bar_chart(
                chart_data.set_index("Class")
            )

        st.markdown("---")

        # =====================================
        # PROJECT STATS
        # =====================================
        st.subheader("📊 Project Statistics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Classes",
            "2"
        )

        c2.metric(
            "Image Size",
            "64x64"
        )

        c3.metric(
            "Framework",
            "Streamlit"
        )

        c4.metric(
            "Model",
            "Scikit-Learn"
        )

    except Exception as e:
        st.error(f"Error : {e}")

# =====================================
# AI WORKFLOW
# =====================================
st.markdown("---")

st.subheader("🧠 AI Workflow")

st.code("""
📤 Upload Image
        ↓
🖼️ Image Processing
        ↓
🔍 Feature Extraction
        ↓
🤖 Scikit-Learn Model
        ↓
📊 Confidence Score
        ↓
✅ Final Prediction
""")

# # =====================================
# # DEVELOPER CARD
# # =====================================
# st.markdown("---")

# st.markdown("""
# <div class="dev-card">

# <h2>👨‍💻 Developed by Richeek Pandey</h2>

# <p>
# AI/ML Enthusiast • Data Science Learner • B.Tech IT
# </p>

# <p>
# <a href="https://www.linkedin.com/in/richeek-pandey-9954783a9" target="_blank">
# 🔗 LinkedIn
# </a>
# &nbsp;&nbsp;&nbsp;
# <a href="https://github.com/richeekpandey07" target="_blank">
# 💻 GitHub
# </a>
# </p>

# </div>
# """, unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
<hr>
🐱🐶 Cat vs Dog Classifier <br>
Built with Streamlit, OpenCV & Scikit-Learn
</div>
""", unsafe_allow_html=True)

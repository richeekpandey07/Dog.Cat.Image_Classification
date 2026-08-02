import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import cv2
import plotly.express as px

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

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model 'cat_dog_model.pkl': {e}")
    st.stop()

IMG_SIZE = 64

# =====================================
# CUSTOM CSS (Fixes Text & Component Visibility)
# =====================================
st.markdown("""
<style>
/* App Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #312e81);
    color: #ffffff;
}

/* Force Contrast on Text Elements */
html, body, [class*="css"], p, label, div, span, h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141E30, #243B55);
}

/* Header Container */
.main-header {
    background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white !important;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}

/* Feature Cards */
.feature1 { background: linear-gradient(135deg, #06b6d4, #3b82f6); padding: 15px; border-radius: 15px; text-align: center; font-weight: bold; }
.feature2 { background: linear-gradient(135deg, #7c3aed, #a855f7); padding: 15px; border-radius: 15px; text-align: center; font-weight: bold; }
.feature3 { background: linear-gradient(135deg, #ec4899, #f43f5e); padding: 15px; border-radius: 15px; text-align: center; font-weight: bold; }
.feature4 { background: linear-gradient(135deg, #f59e0b, #f97316); padding: 15px; border-radius: 15px; text-align: center; font-weight: bold; }

/* Prediction Cards */
.prediction-card {
    background: linear-gradient(135deg, #10b981, #34d399);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    margin-top: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

/* Workflow Card */
.workflow-card {
    background: linear-gradient(135deg, #4338ca, #7c3aed);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

/* Developer Card */
.dev-card {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
}

.dev-card a {
    color: #ffffff !important;
    font-weight: bold;
    text-decoration: underline;
}

/* File Uploader Container Fix */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 15px;
    border-radius: 15px;
    border: 1px dashed #06b6d4;
}

.footer {
    text-align: center;
    color: #cccccc !important;
    padding: 20px 0;
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
🎓 **B.Tech IT Student**  
🤖 **AI / ML Enthusiast**  
📊 **Data Science Learner**  
""")

st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/richeek-pandey-9954783a9)")
st.sidebar.markdown("[💻 GitHub](https://github.com/richeekpandey07)")

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
<div class='main-header'>
    <h1>🐱 VS 🐶</h1>
    <h3>AI Powered Pet Recognition System</h3>
    <p>Machine Learning • Computer Vision • Streamlit</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# FEATURES
# =====================================
st.subheader("✨ Smart Features")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='feature1'>📸 Image Upload</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='feature2'>🤖 AI Prediction</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='feature3'>📊 Confidence Score</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='feature4'>⚡ Fast Processing</div>", unsafe_allow_html=True)

st.markdown("---")

# =====================================
# IMAGE UPLOAD & PREDICTION
# =====================================
st.subheader("📤 Upload Your Pet Image")

uploaded_file = st.file_uploader(
    "Choose a Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col_img, col_pred = st.columns([1, 1])
    
    with col_img:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_pred:
        # Preprocessing Image
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        image_flatten = img_resized.flatten()

        # Model Prediction
        prediction = model.predict([image_flatten])[0]
        
        # Check if model supports probability estimation
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba([image_flatten])[0]
            cat_prob = probability[0]
            dog_prob = probability[1]
        else:
            probability = None

        # Prediction Card
        if prediction == 0:
            st.markdown("<div class='prediction-card'>🐱 CAT DETECTED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction-card'>🐶 DOG DETECTED</div>", unsafe_allow_html=True)

        st.balloons()
        st.toast("Prediction Complete 🚀")

    # =====================================
    # CHART & ANALYTICS
    # =====================================
    if probability is not None:
        st.markdown("---")
        st.subheader("📈 Prediction Analytics Dashboard")

        chart_data = pd.DataFrame({
            "Class": ["🐱 Cat", "🐶 Dog"],
            "Probability": [cat_prob * 100, dog_prob * 100]
        })

        fig = px.bar(
            chart_data,
            x="Class",
            y="Probability",
            text="Probability",
            title="Prediction Confidence Score",
            color="Class",
            color_discrete_sequence=["#06b6d4", "#ec4899"]
        )

        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            height=400,
            xaxis_title="Pet Category",
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 110],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🎯 AI Decision Summary")
        winner_prob = max(cat_prob, dog_prob) * 100
        
        if prediction == 0:
            st.success(f"🐱 The model predicts this image is a **CAT** with **{winner_prob:.2f}%** confidence.")
        else:
            st.success(f"🐶 The model predicts this image is a **DOG** with **{winner_prob:.2f}%** confidence.")

st.markdown("---")

# =====================================
# PROJECT STATS
# =====================================
st.markdown("## 📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#06b6d4,#3b82f6); padding:20px; border-radius:15px; text-align:center;">
        <h3>2</h3>
        <p>Classes</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#7c3aed,#a855f7); padding:20px; border-radius:15px; text-align:center;">
        <h3>64×64</h3>
        <p>Image Size</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#ec4899,#f43f5e); padding:20px; border-radius:15px; text-align:center;">
        <h3>Streamlit</h3>
        <p>Framework</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#f59e0b,#f97316); padding:20px; border-radius:15px; text-align:center;">
        <h3>ML</h3>
        <p>Model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### 🚀 Model Highlights")
a, b, c = st.columns(3)
with a:
    st.success("⚡ Real-Time Prediction")
with b:
    st.success("🤖 Machine Learning Powered")
with c:
    st.success("📸 Image Classification")

st.markdown("---")

# =====================================
# AI WORKFLOW
# =====================================
st.markdown("""
<div class='workflow-card'>
    <h3>🧠 AI Workflow</h3>
    <p>📤 Upload Image → 🖼️ Image Processing → 🔍 Feature Extraction → 🤖 Scikit-Learn Model → 📊 Confidence Analysis → ✅ Final Prediction</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# DEVELOPER CARD
# =====================================
st.markdown("---")

st.markdown("""
<div class="dev-card">
    <h2>👨‍💻 Richeek Pandey</h2>
    <h4>AI/ML Enthusiast • Data Science Learner</h4>
    <p>Building Machine Learning & Data Science Projects</p>
    <p>
        <a href="https://www.linkedin.com/in/richeek-pandey-9954783a9" target="_blank">🔗 LinkedIn</a>
        &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/richeekpandey07" target="_blank">💻 GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
    <hr>
    🚀 Cat vs Dog AI Classifier <br>
    Built with ❤️ using Streamlit, OpenCV & Scikit-Learn <br><br>
    © 2026 Richeek Pandey
</div>
""", unsafe_allow_html=True)

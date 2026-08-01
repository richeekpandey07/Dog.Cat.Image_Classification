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

.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b,#312e81);
}

section[data-testid="stSidebar"]{
background: linear-gradient(180deg,#141E30,#243B55);
}

.main-header{
background: linear-gradient(90deg,#06b6d4,#8b5cf6);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
box-shadow:0px 5px 20px rgba(0,0,0,0.3);
}

.feature1{
background:linear-gradient(135deg,#06b6d4,#3b82f6);
padding:15px;
border-radius:15px;
text-align:center;
color:white;
font-weight:bold;
}

.feature2{
background:linear-gradient(135deg,#7c3aed,#a855f7);
padding:15px;
border-radius:15px;
text-align:center;
color:white;
font-weight:bold;
}

.feature3{
background:linear-gradient(135deg,#ec4899,#f43f5e);
padding:15px;
border-radius:15px;
text-align:center;
color:white;
font-weight:bold;
}

.feature4{
background:linear-gradient(135deg,#f59e0b,#f97316);
padding:15px;
border-radius:15px;
text-align:center;
color:white;
font-weight:bold;
}

.prediction-card{
background:linear-gradient(135deg,#10b981,#34d399);
padding:20px;
border-radius:15px;
text-align:center;
color:white;
font-size:25px;
font-weight:bold;
}

.workflow-card{
background:linear-gradient(135deg,#4338ca,#7c3aed);
padding:20px;
border-radius:15px;
color:white;
}

.dev-card{
background:linear-gradient(135deg,#ff416c,#ff4b2b);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
}

.footer{
text-align:center;
color:white;
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

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("<div class='feature1'>📸 Image Upload</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='feature2'>🤖 AI Prediction</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='feature3'>📊 Confidence Score</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='feature4'>⚡ Fast Processing</div>", unsafe_allow_html=True)

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
if prediction == 0:
    st.markdown("""
    <div class='prediction-card'>
    🐱 CAT DETECTED
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='prediction-card'>
    🐶 DOG DETECTED
    </div>
    """, unsafe_allow_html=True)

st.balloons()
st.toast("Prediction Complete 🚀")
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
        st.markdown("### 🎯 Prediction Summary"
                    winner = max(cat_prob, dog_prob) * 100

         st.info(
         f"Model confidence is {winner:.2f}% for the predicted class."
          )
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
st.markdown("""
<div class='workflow-card'>

<h3>🧠 AI Workflow</h3>

📤 Upload Image <br><br>

⬇️ <br><br>

🖼️ Image Processing <br><br>

⬇️ <br><br>

🔍 Feature Extraction <br><br>

⬇️ <br><br>

🤖 Scikit-Learn Model <br><br>

⬇️ <br><br>

📊 Confidence Analysis <br><br>

⬇️ <br><br>

✅ Final Prediction

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

<p>
Building Machine Learning & Data Science Projects
</p>

<p>
<a href="https://www.linkedin.com/in/richeek-pandey-9954783a9" target="_blank">
🔗 LinkedIn
</a>

&nbsp;&nbsp;&nbsp;&nbsp;

<a href="https://github.com/richeekpandey07" target="_blank">
💻 GitHub
</a>
</p>

</div>
""", unsafe_allow_html=True)
# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">
<hr>

🚀 Cat vs Dog AI Classifier

<br>

Built with ❤️ using Streamlit, OpenCV & Scikit-Learn

<br><br>

© 2026 Richeek Pandey

</div>
""", unsafe_allow_html=True)

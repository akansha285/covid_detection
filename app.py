import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detector",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #f4f9ff, #eef7f8);
    color: #1f2937;
    font-family: 'Segoe UI', sans-serif;
}

/* Hero title */
.hero-box {
    background: linear-gradient(135deg, #0f766e, #2563eb);
    padding: 28px 24px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 1rem;
    opacity: 0.95;
}

/* Card box */
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    margin-top: 10px;
    margin-bottom: 16px;
}

/* Result boxes */
.result-covid {
    background: #fff1f2;
    border-left: 8px solid #dc2626;
    padding: 18px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    color: #991b1b;
    margin-top: 15px;
}

.result-normal {
    background: #ecfdf5;
    border-left: 8px solid #16a34a;
    padding: 18px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    color: #166534;
    margin-top: 15px;
}

/* Small metric cards */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 14px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04);
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 14px;
    margin-top: 25px;
}

/* Button styling */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #0f766e);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-size: 16px;
    font-weight: 700;
    transition: 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 18px rgba(37, 99, 235, 0.25);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def load_covid_model():
    model = tf.keras.models.load_model("model.keras")   
    return model

model = load_covid_model()

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((299, 299))  # keep same as training image size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed, verbose=0)

    prob = float(prediction[0][0])

    if prob > 0.5:
        label = "COVID"
        confidence = prob * 100
    else:
        label = "NORMAL"
        confidence = (1 - prob) * 100

    return label, confidence, prob

# ----------------------------
# Hero Section
# ----------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🩺 COVID-19 Chest X-Ray Detector</div>
    <div class="hero-subtitle">
        Upload a chest X-ray image and get an AI-powered prediction for
        <b>COVID</b> or <b>NORMAL</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Upload Card
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📤 Upload Chest X-Ray")
st.write("Supported formats: **JPG, JPEG, PNG**")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded X-Ray Image", width=320)

    if st.button("🔍 Analyze X-Ray"):
        with st.spinner("Analyzing image with AI model..."):
            label, confidence, raw_prob = predict_image(img)

        st.markdown("### 🧾 Prediction Result")

        if label == "COVID":
            st.markdown(
                f'<div class="result-covid">🦠 Prediction: {label}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-normal">✅ Prediction: {label}</div>',
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Confidence Score")
        st.progress(min(int(confidence), 100))
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Metrics row
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4 style="margin-bottom:6px;">Predicted Class</h4>
                    <div style="font-size:20px; font-weight:700;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4 style="margin-bottom:6px;">Raw Model Output</h4>
                    <div style="font-size:20px; font-weight:700;">{raw_prob:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(
            "⚠️ This prediction is generated by a machine learning model and should be used for educational/demo purposes only, not as a medical diagnosis."
        )

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using <b>Streamlit</b> and <b>TensorFlow</b>
</div>
""", unsafe_allow_html=True)

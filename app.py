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
# Load trained model
# ----------------------------
@st.cache_resource
def load_covid_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_covid_model()

# ----------------------------
# Class labels
# ----------------------------
CLASS_NAMES = ["NORMAL", "COVID"]

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)

    # Since your model uses sigmoid output with Dense(1)
    prob = float(prediction[0][0])

    if prob > 0.5:
        label = "COVID"
        confidence = prob * 100
    else:
        label = "NORMAL"
        confidence = (1 - prob) * 100

    return label, confidence, prob

# ----------------------------
# UI
# ----------------------------
st.title("🩺 COVID-19 Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and the model will predict whether it is **COVID** or **NORMAL**.")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence, raw_prob = predict_image(image)

        st.success("Prediction complete!")

        if label == "COVID":
            st.error(f"🦠 Prediction: **{label}**")
        else:
            st.success(f"✅ Prediction: **{label}**")

        st.info(f"Confidence: **{confidence:.2f}%**")

        # Optional probability display
        st.write("### Prediction Details")
        st.write(f"Raw model output: `{raw_prob:.4f}`")
        st.progress(min(int(confidence), 100))

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with Streamlit and TensorFlow")
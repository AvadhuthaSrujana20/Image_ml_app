import joblib
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Classical ML Image Predictor", page_icon="🖼️")
st.title("🖼️ Image Prediction using Classical ML Pipeline")
st.write("Upload a PNG/JPG image — trained scikit-learn Pipeline predict chesthundi.")

LABELS = {
    0: "Circle",
    1: "Square",
    2: "Triangle",
}

# IKADA PATH PASTE CHEYYALI (project root nunchi relative path)
MODEL_PATH = "artifacts/image_ml_pipeline.joblib"


@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


try:
    pipeline = load_pipeline(MODEL_PATH)
except FileNotFoundError:
    st.error(f"'{MODEL_PATH}' dorakaledu. Mundu `python train.py` run cheyyi.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img_16x16 = img.resize((16, 16))
    st.image(img_16x16, caption="Resized 16x16 Grayscale Image", width=160)

    features = np.asarray(img_16x16, dtype=np.float32).reshape(1, -1)

    if st.button("Predict"):
        pred = int(pipeline.predict(features)[0])
        label_name = LABELS.get(pred, f"Class {pred}")
        st.success(f"Prediction: **{label_name}** (Class {pred})")

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(features)[0]
            conf = float(np.max(probs)) * 100
            st.info(f"Confidence: **{conf:.2f}%**")
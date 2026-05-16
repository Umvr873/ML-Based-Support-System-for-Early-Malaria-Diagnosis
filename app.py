import os
import json
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Malaria AI Diagnostic Support System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    /* =========================
       GLOBAL READABILITY FIXES
       ========================= */
    :root {
        --primary: #8b1e1e;
        --primary-2: #c2410c;
        --text-main: #172033;
        --text-muted: #475569;
        --card-bg: #ffffff;
        --soft-bg: #f8fafc;
        --border: #e5e7eb;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #fff7ed 48%, #f0fdf4 100%);
        color: var(--text-main) !important;
    }

    /* Force Streamlit text to stay readable */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--text-main);
    }

    /* Sidebar readability */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #7f1d1d 0%, #991b1b 55%, #431407 100%);
        border-right: 1px solid rgba(255,255,255,0.15);
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stJson"] *,
    section[data-testid="stSidebar"] pre,
    section[data-testid="stSidebar"] code {
        color: #111827 !important;
        background: #ffffff !important;
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,0.12);
        padding: 0.45rem 0.7rem;
        border-radius: 12px;
        margin-bottom: 0.35rem;
        border: 1px solid rgba(255,255,255,0.14);
    }

    /* Main page container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* =========================
       HEADER
       ========================= */
    .main-header {
        background: linear-gradient(90deg, #7f1d1d, #b91c1c, #ea580c);
        padding: 2.1rem;
        border-radius: 24px;
        color: #ffffff !important;
        box-shadow: 0px 12px 35px rgba(127, 29, 29, 0.22);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.25);
    }

    .main-header h1 {
        font-size: 2.35rem;
        font-weight: 850;
        margin-bottom: 0.45rem;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    .main-header p {
        font-size: 1.08rem;
        color: #fff7ed !important;
        opacity: 1;
        margin-bottom: 0;
        line-height: 1.55;
    }

    /* =========================
       CARDS
       ========================= */
    .info-card, .metric-card {
        background: var(--card-bg);
        padding: 1.25rem;
        border-radius: 20px;
        box-shadow: 0px 8px 26px rgba(15, 23, 42, 0.08);
        border: 1px solid var(--border);
        color: var(--text-main) !important;
    }

    .info-card *, .metric-card * {
        color: var(--text-main) !important;
    }

    .metric-card {
        text-align: center;
        min-height: 125px;
    }

    .metric-card h3 {
        color: var(--text-muted) !important;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }

    .metric-card h2 {
        color: #7f1d1d !important;
        font-size: 2rem;
        font-weight: 800;
    }

    /* =========================
       RESULT CARDS
       ========================= */
    .result-card-positive {
        background: linear-gradient(135deg, #fff1f2, #fee2e2);
        border-left: 8px solid #dc2626;
        padding: 1.45rem;
        border-radius: 20px;
        box-shadow: 0px 8px 28px rgba(220, 38, 38, 0.13);
        color: #450a0a !important;
    }

    .result-card-positive h2,
    .result-card-positive h3,
    .result-card-positive p {
        color: #450a0a !important;
    }

    .result-card-negative {
        background: linear-gradient(135deg, #ecfdf5, #dcfce7);
        border-left: 8px solid #16a34a;
        padding: 1.45rem;
        border-radius: 20px;
        box-shadow: 0px 8px 28px rgba(22, 163, 74, 0.13);
        color: #052e16 !important;
    }

    .result-card-negative h2,
    .result-card-negative h3,
    .result-card-negative p {
        color: #052e16 !important;
    }

    .result-card-invalid {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border-left: 8px solid #f97316;
        padding: 1.45rem;
        border-radius: 20px;
        box-shadow: 0px 8px 28px rgba(249, 115, 22, 0.13);
        color: #431407 !important;
    }

    .result-card-invalid h2,
    .result-card-invalid h3,
    .result-card-invalid p {
        color: #431407 !important;
    }

    /* =========================
       WARNING / ALERTS
       ========================= */
    .warning-box {
        background: #fffbeb;
        border-left: 7px solid #d97706;
        padding: 1rem;
        border-radius: 16px;
        color: #451a03 !important;
        font-size: 0.96rem;
        line-height: 1.55;
        margin-top: 0.7rem;
    }

    .warning-box * {
        color: #451a03 !important;
    }

    /* Native Streamlit info/warning readability */
    div[data-testid="stAlert"] {
        background: #ffffff !important;
        color: var(--text-main) !important;
        border-radius: 14px;
        border: 1px solid var(--border);
    }

    div[data-testid="stAlert"] * {
        color: var(--text-main) !important;
    }

    /* Upload area readability */
    .upload-panel {
        background: linear-gradient(135deg, #fff7ed, #ffffff 55%, #fef2f2);
        border: 2px dashed #dc2626;
        border-radius: 22px;
        padding: 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: 0px 8px 24px rgba(127, 29, 29, 0.08);
        text-align: center;
    }

    .upload-panel h3 {
        color: #7f1d1d !important;
        font-size: 1.2rem;
        margin-bottom: 0.35rem;
        font-weight: 800;
    }

    .upload-panel p {
        color: #475569 !important;
        margin-bottom: 0;
        font-size: 0.95rem;
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        border-radius: 18px !important;
        padding: 1rem !important;
        box-shadow: 0px 7px 22px rgba(15,23,42,0.08) !important;
        border: 2px solid #fee2e2 !important;
    }

    div[data-testid="stFileUploader"] section {
        background: #fff7f7 !important;
        border: 2px dashed #ef4444 !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }

    div[data-testid="stFileUploader"] * {
        color: #111827 !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {
        color: #374151 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #7f1d1d, #dc2626) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 800 !important;
    }

    /* Buttons */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(90deg, #7f1d1d, #dc2626) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        box-shadow: 0px 6px 18px rgba(127, 29, 29, 0.22);
    }

    .stButton button:hover, .stDownloadButton button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    /* Dataframe and chart areas */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        background: #ffffff;
        border-radius: 18px;
        padding: 0.4rem;
        border: 1px solid var(--border);
    }

    /* Captions and small text */
    .stCaptionContainer, .stCaptionContainer * {
        color: #334155 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #334155 !important;
        font-size: 0.9rem;
        background: rgba(255,255,255,0.62);
        border-radius: 16px;
        border: 1px solid rgba(226,232,240,0.9);
    }

    .footer * {
        color: #334155 !important;
    }


    /* =========================
       STREAMLIT TOP BAR + SIDEBAR TOGGLE
       ========================= */

    /* Keep the top Streamlit header clean and readable */
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #111827 !important;
    }

    header[data-testid="stHeader"] * {
        color: #111827 !important;
    }

    /* Target only the top-left sidebar/menu toggle button */
    header[data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"],
    header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"],
    header[data-testid="stHeader"] button[aria-label="Open sidebar"],
    header[data-testid="stHeader"] button[aria-label="Close sidebar"] {
        background-color: #D62828 !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        min-height: 38px !important;
        padding: 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.65) !important;
        box-shadow: 0 6px 18px rgba(214, 40, 40, 0.30) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Make only the sidebar/menu arrow icon white */
    header[data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"] svg,
    header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"] svg,
    header[data-testid="stHeader"] button[aria-label="Open sidebar"] svg,
    header[data-testid="stHeader"] button[aria-label="Close sidebar"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    header[data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"]:hover,
    header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"]:hover,
    header[data-testid="stHeader"] button[aria-label="Open sidebar"]:hover,
    header[data-testid="stHeader"] button[aria-label="Close sidebar"]:hover {
        background-color: #b91c1c !important;
        transform: scale(1.03);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# PATHS
# =========================
MODEL_PATH = os.path.join("model", "best_malaria_mobilenetv2.keras")
MAPPING_PATH = os.path.join("model", "class_mapping.json")
HISTORY_PATH = "prediction_history.csv"
IMG_SIZE = 224


# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_mapping():
    if not os.path.exists(MAPPING_PATH):
        return {"Parasitized": 0, "Uninfected": 1}
    with open(MAPPING_PATH, "r") as f:
        return json.load(f)


model = load_model()
class_mapping = load_class_mapping()


# =========================
# HELPER FUNCTIONS
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_malaria(image: Image.Image):
    processed_img = preprocess_image(image)
    probability = float(model.predict(processed_img, verbose=0)[0][0])

    # Current training mapping:
    # {'Parasitized': 0, 'Uninfected': 1}
    if probability < 0.5:
        label = "Parasitized"
        confidence = (1 - probability) * 100
        parasitized_prob = (1 - probability) * 100
        uninfected_prob = probability * 100
    else:
        label = "Uninfected"
        confidence = probability * 100
        parasitized_prob = (1 - probability) * 100
        uninfected_prob = probability * 100

    return label, confidence, parasitized_prob, uninfected_prob, probability


def validate_blood_smear_image(image: Image.Image):
    """
    Stronger out-of-scope image validation.

    Important:
    This is still a heuristic gate, not a separately trained medical-image validator.
    It rejects obvious non-blood-smear images before the malaria classifier is allowed
    to make a prediction.
    """
    try:
        original_w, original_h = image.size
        aspect_ratio = original_w / max(original_h, 1)

        # Most single-cell malaria dataset images are close to square.
        # This rejects many selfies, documents, landscapes, and camera photos.
        if aspect_ratio < 0.70 or aspect_ratio > 1.45:
            return (
                False,
                "Invalid image detected. The image shape does not look like a single-cell microscopic blood smear image.",
                0
            )

        rgb_img = image.convert("RGB").resize((224, 224))
        img = np.array(rgb_img)

        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))

        if mean_intensity < 35:
            return False, "The uploaded image is too dark for reliable microscopic analysis.", 0

        if mean_intensity > 238:
            return False, "The uploaded image is too bright or almost blank.", 0

        if std_intensity < 18:
            return False, "The uploaded image appears blank or has very little microscopic detail.", 0

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Blood smear images usually have pale background plus pink/purple/blue stain.
        stained_pixels = (
            (saturation > 25) &
            (value > 45) &
            (
                (hue < 18) |                 # red / pink
                (hue > 135) |                # magenta / purple
                ((hue > 85) & (hue < 135))   # blue-purple stain
            )
        )

        stained_ratio = float(np.mean(stained_pixels))
        pale_background_ratio = float(np.mean((saturation < 70) & (value > 120)))

        # Reject images with almost no stain colour or no light microscope-like background.
        if stained_ratio < 0.020:
            return (
                False,
                "Invalid image detected. The uploaded image does not contain enough blood-smear stain colour.",
                0
            )

        if pale_background_ratio < 0.20:
            return (
                False,
                "Invalid image detected. The background does not look like a microscope blood smear field.",
                0
            )

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Edge/texture check.
        edges = cv2.Canny(gray, 40, 140)
        edge_ratio = float(np.mean(edges > 0))

        if edge_ratio < 0.006:
            return (
                False,
                "Invalid image detected. The image does not contain enough cell-like microscopic texture.",
                0
            )

        # Cell-like object check using contours.
        # This helps reject documents, random photos, screenshots, and smooth backgrounds.
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            4
        )

        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cell_like_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120 or area > 18000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            x, y, w, h = cv2.boundingRect(cnt)
            box_ratio = w / max(h, 1)

            if 0.18 <= circularity <= 1.35 and 0.45 <= box_ratio <= 2.20:
                cell_like_count += 1

        # Scoring system.
        validation_score = 0

        if 0.04 <= stained_ratio <= 0.70:
            validation_score += 30
        elif stained_ratio >= 0.02:
            validation_score += 15

        if pale_background_ratio >= 0.35:
            validation_score += 25
        elif pale_background_ratio >= 0.20:
            validation_score += 10

        if edge_ratio >= 0.015:
            validation_score += 20
        elif edge_ratio >= 0.006:
            validation_score += 10

        if cell_like_count >= 1:
            validation_score += 25

        # Strict gate: prediction is blocked unless image looks reasonably like a stained cell image.
        if validation_score < 70 or cell_like_count < 1:
            return (
                False,
                "Invalid image detected. Please upload a clear microscopic blood smear cell image. Random photos, documents, screenshots, X-rays, or non-cell images are not accepted.",
                validation_score
            )

        return True, "Image passed blood-smear validation.", validation_score

    except Exception as e:
        return False, f"Image validation failed: {e}", 0


def find_last_conv_layer_name(model):
    """
    Finds a suitable convolutional feature layer for Grad-CAM.
    This is more robust for saved MobileNetV2/Keras models.
    """
    preferred_layers = [
        "out_relu",
        "Conv_1_relu",
        "Conv_1",
        "block_16_project",
        "block_16_depthwise_relu"
    ]

    for layer_name in preferred_layers:
        try:
            layer = model.get_layer(layer_name)
            if len(layer.output.shape) == 4:
                return layer_name
        except Exception:
            pass

    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue

    return None


def make_gradcam_heatmap(image: Image.Image, model, last_conv_layer_name=None):
    """
    Grad-CAM implementation for MobileNetV2-based Keras model.
    This version fixes the deployed Streamlit error:
    'list indices must be integers or slices, not tuple'
    """
    img_array = preprocess_image(image)

    # Try common MobileNetV2 final convolutional layers first
    possible_layers = [
        "out_relu",
        "Conv_1_relu",
        "Conv_1",
        "block_16_project",
        "block_15_project"
    ]

    if last_conv_layer_name is None:
        for layer_name in possible_layers:
            try:
                layer = model.get_layer(layer_name)
                if len(layer.output.shape) == 4:
                    last_conv_layer_name = layer_name
                    break
            except Exception:
                continue

    # If known names fail, automatically search for the last 4D layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer_name = layer.name
                    break
            except Exception:
                continue

    if last_conv_layer_name is None:
        return None

    try:
        # Use model.outputs[0], not model.output, to avoid nested list issues
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.outputs[0]
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            # Fix for Streamlit/Keras list output issue
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)

        if max_val == 0:
            return None

        heatmap = heatmap / max_val

        if hasattr(heatmap, "numpy"):
            return heatmap.numpy()

        return heatmap

    except Exception as e:
        st.info(f"Grad-CAM debug note: {e}")
        return None
    
def overlay_heatmap(image: Image.Image, heatmap, alpha=0.42):
    image = image.convert("RGB")
    original = np.array(image)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def save_prediction(filename, label, confidence, parasitized_prob, uninfected_prob):
    record = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "prediction": label,
        "confidence": round(confidence, 2),
        "parasitized_probability": round(parasitized_prob, 2),
        "uninfected_probability": round(uninfected_prob, 2)
    }

    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(HISTORY_PATH, index=False)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=85)
    st.title("Malaria AI System")
    st.caption("Real-time diagnostic support using MobileNetV2")

    st.markdown("---")
    st.subheader("Model Information")
    st.write("**Model:** MobileNetV2 Transfer Learning")
    st.write("**Input size:** 224 × 224")
    st.write("**Classes:** Parasitized / Uninfected")
    st.write("**Validation/Test Accuracy:** 94%")
    st.write("**Class mapping:**")
    st.json(class_mapping)

    st.markdown("---")
    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        ["Diagnosis", "Model Performance", "Prediction History", "About System"]
    )


# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="main-header">
        <h1>🩸 ML-Based Support System for Early Malaria Diagnosis</h1>
        <p>Deep learning-powered microscopic blood smear image classification for low-resource healthcare support.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# MODEL FILE CHECK
# =========================
if model is None:
    st.error(
        "Model file not found. Please place `best_malaria_mobilenetv2.keras` inside the `model/` folder."
    )
    st.stop()


# =========================
# PAGE 1: DIAGNOSIS
# =========================
if page == "Diagnosis":
    left, right = st.columns([1.1, 1])

    uploaded_file = None
    image = None
    image_is_valid_for_gradcam = False

    with left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Upload Blood Smear Image")
        st.markdown(
            """
            <div class="upload-panel">
                <h3>📤 Upload Microscopic Blood Smear Image</h3>
                <p>Accepted formats: JPG, JPEG, or PNG. The system now rejects non-microscope images before prediction.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Choose a microscopic blood smear image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown(
            """
            <div class="warning-box">
                <b>Important:</b> This system is a diagnostic support tool only.
                It should not replace laboratory testing, microscopy confirmation,
                rapid diagnostic tests, or medical advice from qualified healthcare professionals.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception:
                image = None
                st.error("The uploaded file could not be opened as a valid image.")

    with right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if uploaded_file is None:
            st.info("Upload an image to begin malaria screening.")

        elif image is None:
            st.error("Invalid upload. Please upload a valid JPG, JPEG, or PNG image.")

        else:
            is_valid_image, validation_message, validation_score = validate_blood_smear_image(image)

            if not is_valid_image:
                st.markdown(
                    f"""
                    <div class="result-card-invalid">
                        <h2>🚫 Invalid Image Detected</h2>
                        <h3>Prediction not performed</h3>
                        <p>{validation_message}</p>
                        <p><b>Validation score:</b> {validation_score}/100</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(
                    "Please upload a valid microscopic blood smear image."
                )

            else:
                image_is_valid_for_gradcam = True

                with st.spinner("Analyzing blood smear image..."):
                    time.sleep(1)
                    label, confidence, parasitized_prob, uninfected_prob, raw_prob = predict_malaria(image)

                if label == "Parasitized":
                    st.markdown(
                        f"""
                        <div class="result-card-positive">
                            <h2>⚠️ Malaria Parasite Detected</h2>
                            <h3>Prediction: {label}</h3>
                            <h3>Confidence: {confidence:.2f}%</h3>
                            <p>The model detected visual patterns associated with parasitized blood cells.
                            Immediate confirmatory testing and clinical consultation are recommended.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-negative">
                            <h2>✅ No Parasite Detected</h2>
                            <h3>Prediction: {label}</h3>
                            <h3>Confidence: {confidence:.2f}%</h3>
                            <p>The image was classified as uninfected by the model.
                            Clinical symptoms should still be considered where malaria is suspected.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.caption(f"Image validation score: {validation_score}/100")

                st.markdown("### Prediction Probabilities")
                prob_df = pd.DataFrame({
                    "Class": ["Parasitized", "Uninfected"],
                    "Probability (%)": [parasitized_prob, uninfected_prob]
                })
                st.bar_chart(prob_df.set_index("Class"))

                save_prediction(
                    uploaded_file.name,
                    label,
                    confidence,
                    parasitized_prob,
                    uninfected_prob
                )

        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None and image is not None and image_is_valid_for_gradcam:
        st.markdown("---")
        st.subheader("Explainable AI: Grad-CAM Visualization")

        heatmap = make_gradcam_heatmap(image, model)
        if heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption="Original Image", use_container_width=True)
            with c2:
                st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
            st.caption(
                "Grad-CAM highlights image regions that influenced the model prediction. "
                "It supports interpretability but should not be treated as medical proof."
            )
        else:
            st.warning("Grad-CAM could not be generated for this model structure.")

    elif uploaded_file is not None and image is not None and not image_is_valid_for_gradcam:
        st.markdown("---")
        st.info("Grad-CAM was not generated because the uploaded image did not pass the blood-smear validation check.")


# =========================
# PAGE 2: MODEL PERFORMANCE
# =========================
elif page == "Model Performance":
    st.subheader("Model Evaluation Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="metric-card"><h3>Accuracy</h3><h2>94%</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>Precision</h3><h2>0.94</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>Recall</h3><h2>0.94</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h3>F1-Score</h3><h2>0.94</h2></div>', unsafe_allow_html=True)

    st.markdown("### Classification Report")

    report_df = pd.DataFrame({
        "Class": ["Parasitized", "Uninfected", "Macro Avg", "Weighted Avg"],
        "Precision": [0.95, 0.93, 0.94, 0.94],
        "Recall": [0.93, 0.95, 0.94, 0.94],
        "F1-Score": [0.94, 0.94, 0.94, 0.94],
        "Support": [2755, 2755, 5510, 5510]
    })

    st.dataframe(report_df, use_container_width=True)

    st.markdown(
        """
        <div class="info-card">
        <h4>Interpretation</h4>
        <p>The model achieved an overall accuracy of 94%. For parasitized images,
        the recall score of 0.93 means the model correctly identified most infected samples.
        This is important because missed malaria cases can delay treatment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# PAGE 3: PREDICTION HISTORY
# =========================
elif page == "Prediction History":
    st.subheader("Prediction History")

    if os.path.exists(HISTORY_PATH):
        history_df = pd.read_csv(HISTORY_PATH)
        st.dataframe(history_df.tail(30), use_container_width=True)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prediction History",
            csv,
            "malaria_prediction_history.csv",
            "text/csv"
        )
    else:
        st.info("No predictions have been recorded yet.")


# =========================
# PAGE 4: ABOUT SYSTEM
# =========================
elif page == "About System":
    st.subheader("About the Project")

    st.markdown(
        """
        <div class="info-card">
        <p>
        This system uses a MobileNetV2 transfer learning model to classify microscopic
        blood smear images as either <b>Parasitized</b> or <b>Uninfected</b>.
        The goal is to provide real-time diagnostic support for early malaria screening,
        especially in low-resource settings.
        </p>

        <h4>System Workflow</h4>
        <ol>
            <li>User uploads a blood smear image.</li>
            <li>The image is resized and preprocessed.</li>
            <li>The trained MobileNetV2 model performs classification.</li>
            <li>The app displays prediction, confidence score, probability chart, and Grad-CAM heatmap.</li>
        </ol>

        <h4>Limitations</h4>
        <p>
        The system was trained on publicly available cell image data and does not include
        patient symptoms, age, parasite density, or laboratory metadata. Therefore, it should
        only be used as a support tool and not as a final medical diagnosis.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="footer">
        Deep Learning-Based Malaria Diagnostic Support System
    </div>
    """,
    unsafe_allow_html=True
)

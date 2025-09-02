import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import warnings

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model(
        r"C:\Users\Sanskar Gupta\OneDrive\Desktop\Deep Learning\MRI CLassification Alzimers\cnn_mri_model.keras"
    )
    ann_model = tf.keras.models.load_model(
        r"C:\Users\Sanskar Gupta\OneDrive\Desktop\Deep Learning\MRI CLassification Alzimers\ann_mri_model.keras"
    )
    return cnn_model, ann_model

cnn_model, ann_model = load_models()

# ==============================
# Detect Input Requirements
# ==============================
cnn_input_size = cnn_model.input_shape[1:3]   # (height, width)
ann_input_shape = ann_model.input_shape       # e.g. (None, 128) or (None, H, W, C)

# Figure out ANN expected vector length if it's flat
if len(ann_input_shape) == 2:
    ann_expected_len = ann_input_shape[1]     # e.g. 128
    ann_is_flat = True
else:
    ann_is_flat = False
    ann_expected_len = None

# ==============================
# Class Names
# ==============================
CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Alzheimer MRI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Alzheimer MRI Disease Classification")
st.write("Upload an MRI brain scan and classify the stage of dementia using AI models.")

# Model selection
model_choice = st.radio(
    "Select Model for Prediction:",
    ("Convolutional Neural Network (CNN)", "Artificial Neural Network (ANN)")
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # ==============================
    # Prediction Logic
    # ==============================
    if model_choice == "Convolutional Neural Network (CNN)":
        target_size = cnn_input_size
        img_resized = cv2.resize(img, target_size)
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = cnn_model.predict(img_array)

    else:  # ANN selected
        if ann_is_flat:
            # Flatten the image and adjust to ANN expected length
            img_resized = cv2.resize(img, cnn_input_size)   # use CNN size as a base
            img_array = img_resized / 255.0
            img_flat = img_array.flatten()  # 1D

            # Adjust vector length to ANN expected input
            if len(img_flat) > ann_expected_len:
                img_flat = img_flat[:ann_expected_len]  # truncate
                warnings.warn(f"Image features truncated to {ann_expected_len} to fit ANN input.")
            elif len(img_flat) < ann_expected_len:
                pad_len = ann_expected_len - len(img_flat)
                img_flat = np.pad(img_flat, (0, pad_len))    # pad with zeros
                warnings.warn(f"Image features padded to {ann_expected_len} to fit ANN input.")

            img_ready = np.expand_dims(img_flat, axis=0)    # shape (1, ann_expected_len)
            prediction = ann_model.predict(img_ready)
        else:
            # ANN expects image-like input (rare, but supported)
            ann_input_size = ann_input_shape[1:3]
            img_resized = cv2.resize(img, ann_input_size)
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = ann_model.predict(img_array)

    # ==============================
    # Post-processing
    # ==============================
    if isinstance(prediction, list):
        prediction = prediction[0]

    prediction = np.array(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    st.markdown(f"### 🩺 Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
    st.success("Prediction completed successfully ✅")
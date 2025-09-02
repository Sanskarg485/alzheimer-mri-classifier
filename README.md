# 🧠 Alzheimer MRI Classifier

A deep learning–powered web application to classify stages of dementia from brain MRI scans using two AI models: a **Convolutional Neural Network (CNN)** and an **Artificial Neural Network (ANN)**.

This project was built with **TensorFlow**, **OpenCV**, and **Streamlit**, making it easy to run locally or deploy to the cloud.

---

## 🚀 Features

- Upload an MRI brain scan and get classification results instantly
- Select between CNN and ANN models for prediction
- Confidence score for prediction
- Clean and interactive web UI with Streamlit
- Easily extendable with your own models and classes

---

## 🧩 Models

Two pre-trained models are included:

1. **CNN Model** (`cnn_mri_model.keras`) — handles image-based input directly.
2. **ANN Model** (`ann_mri_model.keras`) — handles either flattened image vectors or image-like tensors depending on training setup.

> 💡 Both models are expected to output 4 classes:
>
> - Mild Demented
> - Moderate Demented
> - Non Demented
> - Very Mild Demented

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/alzheimer-mri-classifier.git
cd alzheimer-mri-classifier
---
## 🧪 File Structure
.
├── app.py                  # Main Streamlit web app
├── cnn_mri_model.keras     # CNN model file (not in repo by default)
├── ann_mri_model.keras     # ANN model file (not in repo by default)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation



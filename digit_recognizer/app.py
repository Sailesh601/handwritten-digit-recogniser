# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import fitz  # PyMuPDF
import io

# Load trained model
model = tf.keras.models.load_model('digit_model.h5')

st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a **28x28 handwritten digit image** (PNG, JPG, or PDF) to get a prediction.")

uploaded_file = st.file_uploader("Choose a digit image or PDF...", type=["png", "jpg", "jpeg", "pdf"])

def preprocess_image(image: Image.Image):
    image = image.convert("L").resize((28, 28))  # Convert to grayscale and resize
    img_array = np.array(image)

    # Invert if background is white and digit is dark
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dims
    return img_array

if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        st.write("📄 PDF uploaded. Extracting first page...")
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        st.image(img, caption="Extracted Image from PDF", width=200)
    else:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=200)

    st.write("🔍 Processing image and predicting...")
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    pred_class = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{pred_class}**")

    # Optional: Show all class probabilities
    st.write("📊 Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"Digit {i}: {prob:.4f}")

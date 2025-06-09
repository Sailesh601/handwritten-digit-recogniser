# 🧠 Handwritten Digit Recognizer (Streamlit + TensorFlow)

This is a web-based application that recognizes **handwritten digits (0–9)** using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset. It allows users to upload handwritten digit images (PNG, JPG, or PDF) and predicts the digit using a trained deep learning model.

---

## 🔧 Tech Stack

- **Streamlit** – for building the web UI
- **TensorFlow / Keras** – for building and training the CNN model
- **PyMuPDF (fitz)** – to extract images from PDFs
- **Pillow (PIL)** – for image preprocessing
- **NumPy** – for numerical operations

---

## 📁 Project Structure
digit_recognizer/
├── app.py # Streamlit web application
├── train.py # Script to train and save the model
├── digit_model.h5 # Trained CNN model
├── README.md # Project documentation (this file)


---

## 🎓 Model Training (`train.py`)

This script trains a CNN on the MNIST dataset and saves the model as `digit_model.h5`.

### ▶️ Run the training:

python train.py
🌐 Run the App (app.py)
✅ Step-by-step:
#Install dependencies

pip install streamlit tensorflow pillow pymupdf

#Launch the Streamlit app

streamlit run app.py

import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import io

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# --- Google Drive Model Download ---
MODEL_PATH = "plantdisease.h5"
DRIVE_FILE_ID = "YOUR_FILE_ID"  # Replace with your Google Drive file ID
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# --- Configuration ---
IMG_SIZE = (224, 224) 

# Class names
CLASS_NAMES = [
    'Apple___Black_rot', 'Apple___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Common_rust', 'Grape___Black_rot',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_mite',
    'Tomato___Target_Spot', 'Tomato___healthy'
]

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.info(f"Please ensure '{MODEL_PATH}' is in the same directory.")
        return None

model = load_model()

# --- Prediction Function ---
def model_predict(img):
    if not model:
        raise Exception("Model is not loaded.")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img_resized = img.resize(IMG_SIZE)
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if img_array.shape != (1, IMG_SIZE[0], IMG_SIZE[1], 3):
        raise ValueError(f"Preprocessed image has incorrect shape: {img_array.shape}")

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    predicted_class = CLASS_NAMES[predicted_class_index]
    
    return predicted_class, confidence

# --- Streamlit App ---
def main():
    st.title("🌿 AI-Powered Plant Disease Classifier")
    st.markdown("---")
    
    if model is None:
        return

    st.subheader("Upload a Leaf Image for Diagnosis")
    
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG file...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown("---")

        with st.spinner("Analyzing image..."):
            try:
                predicted_class, confidence = model_predict(image)
                
                class_display = predicted_class.replace('___', ' - ')
                confidence_percent = confidence * 100
                
                st.subheader("✅ Prediction Complete")
                
                if "healthy" in predicted_class.lower():
                    st.success(f"**Diagnosis: {class_display}**")
                    st.balloons()
                    st.write(f"The model is {confidence_percent:.2f}% confident.")
                else:
                    st.warning(f"**Diagnosis: {class_display}**")
                    st.write(f"The model is {confidence_percent:.2f}% confident.")
                    st.error("This suggests a disease. Please seek agricultural advice.")

                st.info(f"**Note:** The image was resized to {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels for the model.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Ensure the uploaded file is a valid image.")

if __name__ == "__main__":
    main()

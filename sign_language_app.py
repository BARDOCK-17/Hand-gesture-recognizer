import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import tempfile
import base64

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Class names for Sign Language MNIST dataset
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y']

# Define a function to preprocess the webcam image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to load an image and encode it to base64
def get_base64_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Streamlit app
st.set_page_config(layout="wide")

# Load and encode the background image
background_image_path = 'bg2.png'
background_image_base64 = get_base64_image(background_image_path)

# Custom CSS for styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{background_image_base64});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stButton>button {{
        background-color: #f1c27d;
        color: #8d5524;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .css-1v3fvcr {{
        background-color: #f1c27d;
        color: #8d5524;
    }}
    /* Top bar customization */
    .css-1rs6os {{
        background-color: #f1c27d;
    }}
    .css-1n54oe8 {{
        background-color: #f1c27d;
    }}
    .css-1l5l4nn {{
        color: #8d5524;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Button and webcam setup
col1, col2 = st.columns([8, 2])
with col2:
    st.write("")  # Empty space to align button
    if st.button('Start Recognition'):
        # Set up webcam capture
        cap = cv2.VideoCapture(0)

        # Create a temporary file to save webcam images
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display the webcam image
                st.image(frame, channels="BGR", use_column_width=True)
                
                # Preprocess the image and make prediction
                processed_image = preprocess_image(frame)
                prediction = model.predict(processed_image)
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]

                st.write(f"Prediction: {class_label}")

                # Save image to temporary file
                cv2.imwrite(temp_file.name, frame)

                # Check for exit condition
                if st.button('Stop Recognition'):
                    break

        cap.release()
        cv2.destroyAllWindows()

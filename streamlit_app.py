import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, ERROR

import warnings
warnings.filterwarnings("ignore")  # Optional: suppress Python warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Load the model
model = tf.keras.models.load_model('traffic_sign_cnn_medium.keras')

# Load label mapping
label_map = pd.read_csv('label_names.csv')
label_dict = dict(zip(label_map['ClassId'], label_map['SignName']))

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction
def predict_and_display(img):
    st.image(img, caption='Selected Image', width=250)
    processed = preprocess_image(img)
    pred = model.predict(processed)
    class_id = np.argmax(pred)
    confidence = np.max(pred).astype('float')
    st.markdown(f"### Prediction: `{label_dict[class_id]}`")
    st.write(f"Confidence: {confidence*100:.3f}%")
    st.progress(confidence)

# App layout
st.title("🚦 Traffic Sign Classifier")
st.write("Upload a traffic sign image or select from examples to see the prediction.")

mode = st.sidebar.radio("Choose input method", ['Upload Your Own Image', 'Use Example Image'])

if mode == 'Upload Your Own Image':
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        predict_and_display(img)

else:
    # Show list of example images
    example_dir = "examples"
    example_images = [f for f in os.listdir(example_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_example = st.selectbox("Choose an example image", example_images)
    if selected_example:
        img = Image.open(os.path.join(example_dir, selected_example))
        predict_and_display(img)

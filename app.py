import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
def load_model():
    model = tf.keras.models.load_model('autoencoder_model.keras')
    return model

# Function to preprocess custom images for prediction
def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess a single image for the autoencoder.
    - Resize the image to the target size.
    - Normalize pixel values to the range [0, 1].
    """
    image = img_to_array(image).astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to apply the grayscale effect
def apply_red_effect(image_array, model):
    # Pass the image through the encoder layers only
    encoder_model = Sequential(model.layers[:4])  # Select encoder layers
    encoded_image = encoder_model.predict(image_array)

    # Pass the encoded image through the decoder
    decoder_model = Sequential(model.layers[4:])  # Select decoder layers
    decoded_image = decoder_model.predict(encoded_image)

    # Apply a red filter by keeping the red channel and zeroing out others
    filtered_image = decoded_image[0].copy()  # Take the first (and only) image in the batch
    filtered_image[..., 1] = 0  # Zero out green channel
    filtered_image[..., 2] = 0  # Zero out blue channel

    return filtered_image

def apply_grayscale(image_array, model):
    # Pass the image through the encoder layers only
    encoder_model = Sequential(model.layers[:4])  # Select encoder layers
    encoded_image = encoder_model.predict(image_array)

    # Pass the encoded image through the decoder
    decoder_model = Sequential(model.layers[4:])  # Select decoder layers
    decoded_image = decoder_model.predict(encoded_image)

    # Convert the decoded image to grayscale using the standard formula
    grayscale_image = np.dot(decoded_image[0][..., :3], [0.299, 0.587, 0.114])  # Weighted sum of RGB channels
    grayscale_image = np.expand_dims(grayscale_image, axis=-1)  # Add the grayscale channel back

    # Repeat the grayscale channel across the 3 channels to make it (height, width, 3)
    grayscale_image = np.repeat(grayscale_image, 3, axis=-1)

    return grayscale_image

# Function to apply the X-ray effect using the autoencoder
def apply_xray_effect(image_array, model):
    # Pass the image through the encoder layers only
    encoder_model = Sequential(model.layers[:4])  # Encoder layers
    encoded_image = encoder_model.predict(image_array)

    # Invert the encoded image to simulate X-ray effect
    xray_effect = 1 - encoded_image  # Invert intensities

    # Pass the encoded X-ray effect through the decoder
    decoder_model = Sequential(model.layers[4:])  # Decoder layers
    xray_image = decoder_model.predict(xray_effect)
    return xray_image

# Streamlit application
def run_app():
    st.title("Image Effects with Autoencoder")

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the pre-trained model
        model = load_model()

        # Preprocess the image for prediction
        image_array = preprocess_image(image)

        # Create columns for displaying the effects
        col1, col2 = st.columns(2)

        # Display loader while effects are being applied
        with st.spinner('Applying effects...'):
            with col1:
                # Apply Grayscale effect
                transformed_image_grayscale = apply_grayscale(image_array, model)
                st.image(transformed_image_grayscale, caption="Grayscale Effect", use_column_width=True)

                # Apply Red effect
                transformed_image_red = apply_red_effect(image_array, model)
                st.image(transformed_image_red, caption="Red Effect", use_column_width=True)

            with col2:
                # Apply X-ray effect
                transformed_image_xray = apply_xray_effect(image_array, model)
                st.image(transformed_image_xray[0], caption="X-ray Effect", use_column_width=True)

if __name__ == "__main__":
    run_app()
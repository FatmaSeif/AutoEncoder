import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import cv2

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
    - Remove alpha channel if present.
    """
    # Convert image to array
    image_array = img_to_array(image).astype('float32') / 255.0  # Normalize

    # If the image has an alpha channel (4 channels), remove it
    if image_array.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
        image_array = image_array[..., :3]  # Keep only the RGB channels

    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

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

def apply_noise_effect(image_array, model, noise_type='gaussian', noise_level=0.1):
    """
    Add noise to the input image and then reconstruct using the autoencoder.
    
    Arguments:
    - image_array: The image to which noise is applied (shape: (1, height, width, channels)).
    - model: The trained autoencoder model for reconstruction.
    - noise_type: Type of noise to add ('gaussian', 'salt_and_pepper').
    - noise_level: The standard deviation for the Gaussian noise or the fraction for salt-and-pepper noise.
    
    Returns:
    - noisy_image: The image after noise has been added.
    - reconstructed_image: The image after being processed by the autoencoder.
    """
    
    # Adding noise to the image
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0.0, scale=noise_level, size=image_array.shape)
        noisy_image = np.clip(image_array + noise, 0, 1)  # Ensure values are in range [0, 1]
        
    elif noise_type == 'salt_and_pepper':
        noisy_image = image_array.copy()
        # Generate random salt-and-pepper noise
        salt_pepper_noise = np.random.rand(*image_array.shape)
        noisy_image[salt_pepper_noise < noise_level / 2] = 0  # Pepper noise (black)
        noisy_image[salt_pepper_noise > 1 - noise_level / 2] = 1  # Salt noise (white)
    
    # Pass the noisy image through the autoencoder to reconstruct
    reconstructed_image = model.predict(noisy_image)
    
    return noisy_image, reconstructed_image

def apply_rgb(image_array, model):
    # Pass the image through the encoder layers only
    encoder_model = Sequential(model.layers[:4])  # Select encoder layers
    encoded_image = encoder_model.predict(image_array)

    # Pass the encoded image through the decoder
    decoder_model = Sequential(model.layers[4:])  # Select decoder layers
    decoded_image = decoder_model.predict(encoded_image)

    # Apply RGB split filter: 1/3 Red, 1/3 Green, 1/3 Blue
    filtered_image = decoded_image[0].copy()  # Take the first (and only) image in the batch
    
    # Get image dimensions
    height, width, _ = filtered_image.shape
    
    # Divide the image into thirds for red, green, and blue
    # Left third - Red channel
    filtered_image[:, :width // 3, 1:] = 0  # Zero out green and blue in left third (Red)
    
    # Middle third - Green channel
    filtered_image[:, width // 3:2 * (width // 3), ::2] = 0  # Zero out red and blue in middle third (Green)
    
    # Right third - Blue channel
    filtered_image[:, 2 * (width // 3):, :2] = 0  # Zero out red and green in right third (Blue)

    return filtered_image

def compress_and_reconstruct(image_array, model):
    # Compress and reconstruct
    reconstructed = model.predict(image_array)
    return reconstructed

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

def detect_anomalies(image_array, model):
    # Pass the image through the model (autoencoder) for reconstruction
    reconstructed_image = model.predict(image_array)
    
    # Ensure the two images (input and reconstructed) have the same size
    if image_array.shape != reconstructed_image.shape:
        reconstructed_image = cv2.resize(reconstructed_image[0], (image_array.shape[2], image_array.shape[1]))
        reconstructed_image = np.expand_dims(reconstructed_image, axis=0)
    
    # Calculate anomaly map by comparing the original and reconstructed images
    anomaly_map = np.abs(image_array - reconstructed_image)

    return anomaly_map

import cv2
import numpy as np

# Streamlit application
def run_app():
    st.title("Image Effects with Autoencoder")

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image')

        # Load the pre-trained model
        model = load_model()

        # Preprocess the image for prediction
        image_array = preprocess_image(image)

        # Create columns for displaying the effects
        col1, col2 = st.columns(2)

        noisy_image, reconstructed_image = apply_noise_effect(image_array, model)


        # Display loader while effects are being applied
        with st.spinner('Applying effects...'):
            with col1:
                # Apply Grayscale effect
                transformed_image_grayscale = apply_grayscale(image_array, model)
                st.image(transformed_image_grayscale, caption="Grayscale Image")

                # Apply Red effect
                transformed_image_red = apply_red_effect(image_array, model)
                st.image(transformed_image_red, caption="Red Image")

                # Apply Noise effect
                st.image(noisy_image, caption="Noise Image")

                transformed_anomaly = detect_anomalies(image_array, model)
                st.image(transformed_anomaly, caption="Anomalies Image")

            with col2:
                # Apply X-ray effect
                transformed_image_xray = apply_xray_effect(image_array, model)
                st.image(transformed_image_xray[0], caption="X-ray Image")

                # Apply RGB effect
                transformed_image_rgb = apply_rgb(image_array, model)
                st.image(transformed_image_rgb, caption="RGB Image")

                st.image(reconstructed_image, caption="Denoise Image")

                # Apply Compressed effect
                transformed_image_reconstruct = compress_and_reconstruct(image_array, model)
                st.image(transformed_image_reconstruct, caption="Compressed Image")

if __name__ == "__main__":
    run_app()

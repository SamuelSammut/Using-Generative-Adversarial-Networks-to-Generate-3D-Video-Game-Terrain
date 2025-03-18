import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensuring output directory exists
os.makedirs('../inference_outputs', exist_ok=True)

# Load the trained generator
generator = load_model('saved_models/generator_model.h5', compile=False)

# Set noise dimension
noise_dim = 100

def generate_full_resolution_image(generator, noise_dim, output_filename):
    # Generate one large image
    noise = tf.random.normal([1, noise_dim])
    generated_image = generator(noise, training=False)

    # Convert from [-1,1] to [0,1]
    generated_image = (generated_image.numpy()[0] + 1) / 2.0

    # Extract RGB and DEM
    rgb_image = generated_image[:, :, :3]
    dem_image = generated_image[:, :, 3]

    # Save images
    plt.imsave(os.path.join('../inference_outputs', f'{output_filename}_rgb.png'), rgb_image)
    plt.imsave(os.path.join('../inference_outputs', f'{output_filename}_dem.png'), dem_image, cmap='gray')

# Generate the final full-resolution output
generate_full_resolution_image(generator, noise_dim, 'final_inference_output')

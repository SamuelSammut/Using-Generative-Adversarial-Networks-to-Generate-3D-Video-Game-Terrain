import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim(real_folder, generated_folder):
    """Computes SSIM between real and generated images."""
    real_images = sorted(os.listdir(real_folder))
    gen_images = sorted(os.listdir(generated_folder))

    total_ssim = 0
    num_images = min(len(real_images), len(gen_images))

    for i in range(num_images):
        real_img = cv2.imread(os.path.join(real_folder, real_images[i]))
        gen_img = cv2.imread(os.path.join(generated_folder, gen_images[i]))

        # Resize both images to the same size
        real_img = cv2.resize(real_img, (299, 299))
        gen_img = cv2.resize(gen_img, (299, 299))

        # Convert to grayscale for SSIM computation
        real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        ssim_value = ssim(real_gray, gen_gray)
        total_ssim += ssim_value

    avg_ssim = total_ssim / num_images
    print(f"✅ Average SSIM Score: {avg_ssim:.3f}")

compute_ssim("preprocessed_data_images/rgb/resized", "terrain_outputs/rgb/resized")

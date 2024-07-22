#Gaussian Averaging Filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def apply_gaussian_filter(image, kernel_size=(3, 3), sigma=3.0):
    # Apply a Gaussian filter using OpenCV's GaussianBlur function
    filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)

    return filtered_image

def calculate_psnr(original, noisy, filtered):
    # Calculate PSNR for both noisy and filtered images compared to the original
    psnr_noisy = peak_signal_noise_ratio(original, noisy)
    psnr_filtered = peak_signal_noise_ratio(original, filtered)

    return psnr_noisy, psnr_filtered

# Load the noisy image
image_path = '/content/DSP-2.jpg'  # Replace with the path to your noisy image
noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if noisy_image is None:
    print(f"Error: Unable to load the image at path {image_path}")
else:
    # Apply the Gaussian filter
    filtered_image = apply_gaussian_filter(noisy_image)

    # Calculate PSNR values
    psnr_noisy, psnr_filtered = calculate_psnr(noisy_image, noisy_image, filtered_image)

    # Display the original, noisy, and filtered images
    plt.subplot(131), plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

    plt.show()

    # Print PSNR values
    print(f"PSNR (Noisy): {psnr_noisy:.2f} dB")
    print(f"PSNR (Filtered): {psnr_filtered:.2f} dB")

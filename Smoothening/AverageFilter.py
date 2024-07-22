#Simple Average Filter
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_average_filter(image):
    # Define a 3x3 simple average filter
    kernel = np.ones((3, 3), np.float32) / 225

    # Apply the filter using OpenCV's filter2D function
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

def calculate_psnr(original_image, noisy_image, filtered_image):
    # Calculate mean squared error between original and filtered images
    mse_filtered = np.mean((original_image - filtered_image)**2)

    # Calculate PSNR
    psnr = 20 * np.log10(255 / np.sqrt(mse_filtered))

    return psnr

# Load the noisy image
image_path = '/content/DSP-2.jpg'  # Replace with the path to your noisy image
noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if noisy_image is None:
    print(f"Error: Unable to load the image at path {image_path}")
else:
    # Apply the 3x3 simple average filter
    filtered_image = apply_average_filter(noisy_image)

    # Load the original image for PSNR calculation
    original_image_path = '/content/DSP-2.jpg'  # Replace with the path to your original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the original image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the original image at path {original_image_path}")
    else:
        # Calculate and print PSNR value
        psnr_value = calculate_psnr(original_image, noisy_image, filtered_image)
        print(f"PSNR Value: {psnr_value:.2f} dB")

        # Display the original, noisy, and filtered images
        plt.subplot(131), plt.imshow(original_image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(132), plt.imshow(filtered_image, cmap='gray')
        plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

        plt.show()

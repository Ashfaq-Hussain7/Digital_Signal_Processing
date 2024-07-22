#Laplacian Filter for Sharpening

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sharpen_image_custom(original_image, alpha=1.5):
    # Define a custom Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Apply the custom Laplacian filter
    laplacian_custom = cv2.filter2D(original_image, cv2.CV_64F, laplacian_kernel)

    # Find the discontinuities (edges)
    edges = np.abs(laplacian_custom)

    # Convert the Laplacian edges to the same data type as the original image
    edges = np.uint8(edges)

    # Blend the original image with the edges to sharpen it
    sharpened_image = cv2.addWeighted(original_image, 1.0, edges, alpha, 0)

    return original_image, edges, sharpened_image

# Load the input image
image_path = '/content/DSP-5.webp'  # Replace with the path to your image
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if original_image is None:
    print(f"Error: Unable to load the image at path {image_path}")
else:
    # Apply custom Laplacian filter and blend to sharpen the image
    original, edges_custom, sharpened_image_custom = sharpen_image_custom(original_image)

    # Display the original image
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.show()

    # Display the edges obtained with custom Laplacian
    plt.imshow(edges_custom, cmap='gray')
    plt.title('Edges (Discontinuities) - Custom Laplacian')
    plt.show()

    # Display the sharpened image obtained with custom Laplacian
    plt.imshow(sharpened_image_custom, cmap='gray')
    plt.title('Sharpened Image - Custom Laplacian')
    plt.show()


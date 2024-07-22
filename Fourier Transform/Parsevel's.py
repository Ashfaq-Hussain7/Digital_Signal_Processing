#Program to Implement Parsevel's energy theorem

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure the image path is correct
image_path = '/content/DSP-5.webp'

# Read the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print(f"Failed to load image from {image_path}")
else:
    # Normalize the image to [0, 1]
    image = image / 255.0

    # Flatten the image to a 1D array
    f = np.array(image).flatten()

    # Apply Fast Fourier Transform (FFT)
    F = np.fft.fft(f)

    # Calculate energy in the spatial domain
    space_energy = np.sum(np.abs(f)**2)

    # Calculate energy in the frequency domain
    freq_energy = np.sum(np.abs(F)**2) / f.size

    # Print the energies
    print(f"Energy in the spatial domain: {space_energy}")
    print(f"Energy in the frequency domain: {freq_energy}")

    # Verify Parseval's theorem
    print(f"Parseval's theorem holds: {np.isclose(space_energy,
    freq_energy)}")

    # Display the original image in the spatial domain
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.show()

    # Compute the discrete Fourier Transform of the image
    fourier = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)

    # Calculate the magnitude spectrum of the Fourier Transform
    magnitude_spectrum = 20 * np.log(cv2.magnitude(fourier_shift[:,:,0],
    fourier_shift[:,:,1]))

    # Scale the magnitude spectrum for display
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255,
    cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Display the magnitude spectrum in the Fourier domain
    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.show()
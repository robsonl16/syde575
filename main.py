from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skimage

from lab1.run import run_lab_1
from lab2.run import run_lab_2

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"


if __name__ == "__main__":
    # run_lab_1()
    # run_lab_2()

    # Test image for Part 2
    f = np.zeros((256, 256))
    f[:, 107:148] = 1

    # Rotate the image by 45 degrees
    rotated_f = skimage.transform.rotate(f, 45)

    # Use 2D fourier transform, returns complex array
    fourier_transformed = np.fft.fft2(f)
    rotated_fourier_transformed = np.fft.fft2(rotated_f)
    # Need to use this function to shift the zero frequency component to the center of the spectrum
    # Not sure why the first fourier transform is not centered, but this is what the documentation says to do
    fourier_transformed_shifted = np.fft.fftshift(fourier_transformed)
    rotated_fourier_transformed_shifted = np.fft.fftshift(rotated_fourier_transformed)

    # Convert the complex numbers to magnitude spectrum
    magnitude_spectrum = np.abs(fourier_transformed_shifted)
    magnitude_spectrum_rotated = np.abs(rotated_fourier_transformed_shifted)

    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.imshow(rotated_f, cmap="gray")
    plt.show()
    plt.imshow(magnitude_spectrum_rotated, cmap="gray")
    plt.show()

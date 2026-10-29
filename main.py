from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color, util

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
    rotated_f = transform.rotate(f, 45)

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

    plt.imshow(f, cmap="gray")
    plt.show()
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.show()
    plt.imshow(rotated_f, cmap="gray")
    plt.show()
    plt.imshow(magnitude_spectrum_rotated, cmap="gray")
    plt.show()

    # Load Lena image
    lena = io.imread(LENA_ASSET_PATH)
    lena_gray = color.rgb2gray(np.array(lena))

    # Fourier transform
    fourier_transformed_lena = np.fft.fft2(lena_gray)

    # Shift to center
    fourier_transformed_lena_shifted = np.fft.fftshift(fourier_transformed_lena)

    # Get the magnitude and phase
    fourier_transformed_lena_magnitude = np.abs(fourier_transformed_lena_shifted)
    fourier_transformed_lena_phase = (
        fourier_transformed_lena_shifted / fourier_transformed_lena_magnitude
    )

    print(fourier_transformed_lena_magnitude)
    print(fourier_transformed_lena_phase)

    plt.imshow(fourier_transformed_lena_magnitude, cmap="gray")
    plt.show()
    plt.imshow(np.angle(fourier_transformed_lena_phase), cmap="gray")
    plt.show()

    # Inverse Fourier transform to reconstruct the image from amplitude and phase
    unshifted_magnitude = np.fft.ifftshift(fourier_transformed_lena_magnitude)
    unshifted_phase = np.fft.ifftshift(fourier_transformed_lena_phase)
    inverse_amplitude = np.abs(np.fft.ifft2(unshifted_magnitude))
    inverse_phase = np.abs(np.fft.ifft2(unshifted_phase))

    plt.imshow(lena_gray, cmap="gray")
    plt.show()
    plt.imshow(np.log1p(inverse_amplitude), cmap="gray")
    plt.show()
    plt.imshow(inverse_phase, cmap="gray")
    plt.show()

# Part 2: Noise Reduction in the Frequency Domain:
gaussian_noise_lena = util.random_noise(lena_gray, mode="gaussian", mean=0, var=0.005)

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color, util
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter

from lab1.run import run_lab_1
from lab2.run import run_lab_2


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"
NOISY_IMAGE_ASSET_PATH = ASSETS_DIR / "frequnoisy.tif"


def spectrum(image):
    return np.fft.fftshift(np.fft.fft2(image))


def show(images, titles, cmap="gray", size=4, vrange=None):
    # Plot a row of images side by side, each with a title.
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    axes = np.atleast_1d(axes)
    for ax, im, t in zip(axes, images, titles):
        if vrange is None:
            ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(im, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run_lab_1()
    # run_lab_2()

    # Test image for Part 2
    f = np.zeros((256, 256))
    f[:, 107:148] = 1

    # Rotate the image by 45 degrees
    rotated_f = transform.rotate(f, 45)

    # Use 2D fourier transform, returns complex array
    fourier_transformed = spectrum(f)
    rotated_fourier_transformed = spectrum(rotated_f)

    # Convert the complex numbers to magnitude spectrum

    show(
        [f, np.abs(fourier_transformed)],
        [
            "Test image f",
            "Fourier spectrum",
        ],
    )
    show(
        [rotated_f, np.abs(rotated_fourier_transformed)],
        ["Test image rotated 45°", "Fourier spectrum of rotated image"],
    )

    # Load Lena image
    lena = io.imread(LENA_ASSET_PATH)
    lena_gray = color.rgb2gray(np.array(lena))

    # Fourier transform
    fourier_transformed_lena = np.fft.fft2(lena_gray)

    # Get the magnitude and phase
    fourier_transformed_lena_magnitude = np.abs(fourier_transformed_lena)
    fourier_transformed_lena_phase = (
        fourier_transformed_lena / fourier_transformed_lena_magnitude
    )

    reconstructed_amplitude = np.fft.ifft2(fourier_transformed_lena_magnitude)
    reconstructed_phase = np.fft.ifft2(fourier_transformed_lena_phase)

    show(
        [
            lena_gray,
            np.log1p(np.abs(reconstructed_amplitude)),
            np.abs(reconstructed_phase),
        ],
        [
            "Original (grayscale)",
            "Amplitude reconstructed",
            "Phase reconstructed",
        ],
    )

    # Part 2-1: Noise Reduction in the Frequency Domain:
    # I'm not sure if these plots look right

    gaussian_noise_lena = util.random_noise(
        lena_gray, mode="gaussian", mean=0, var=0.005
    )
    transformed_gaussian_noise_lena = spectrum(gaussian_noise_lena)
    transformed_lena = spectrum(lena_gray)

    show(
        [
            np.log1p(np.abs(transformed_lena)),
            np.log1p(np.abs(transformed_gaussian_noise_lena)),
        ],
        ["Original Lena", "Gaussian noise Lena"],
    )

    # Part 2-2:

    r = 60
    h = disk(r)  # shape (2r+1, 2r+1)

    rows, cols = lena_gray.shape  # filter must match the image size
    h_freq = np.zeros((rows, cols))

    cy, cx = rows // 2, cols // 2
    h_freq[cy - r : cy + r + 1, cx - r : cx + r + 1] = h

    plt.imshow(h_freq, cmap="gray")
    plt.show()

    filtered_lena_fourier = h_freq * transformed_gaussian_noise_lena
    filtered_lena = np.clip(
        np.fft.ifft2(np.fft.ifftshift(filtered_lena_fourier)).real, 0, 1
    )
    show(
        [lena_gray, gaussian_noise_lena, filtered_lena],
        ["Original Lena", "Gaussian noise Lena", "Filtered Lena"],
    )

    r = 20
    h = disk(r)  # shape (2r+1, 2r+1)

    rows, cols = lena_gray.shape  # filter must match the image size
    h_freq = np.zeros((rows, cols))

    cy, cx = rows // 2, cols // 2
    h_freq[cy - r : cy + r + 1, cx - r : cx + r + 1] = h

    filtered_lena_fourier = h_freq * transformed_gaussian_noise_lena
    filtered_lena = np.clip(
        np.fft.ifft2(np.fft.ifftshift(filtered_lena_fourier)).real, 0, 1
    )
    show(
        [lena_gray, gaussian_noise_lena, filtered_lena],
        ["Original Lena", "Gaussian noise Lena", "Filtered Lena"],
    )

    # Build a Gaussian low-pass filter on the shifted frequency grid.
    gaussian_kernel = np.zeros_like(lena_gray)
    gaussian_kernel[rows // 2, cols // 2] = 1
    gaussian_kernel = gaussian_filter(gaussian_kernel, sigma=60, mode="wrap")
    gaussian_kernel /= gaussian_kernel.max()

    gaussian_filtered_spectrum = gaussian_kernel * transformed_gaussian_noise_lena
    gaussian_filtered_lena = np.fft.ifft2(
        np.fft.ifftshift(gaussian_filtered_spectrum)
    ).real

    show([gaussian_kernel], ["Gaussian frequency filter (sigma=60)"])
    show(
        [lena_gray, gaussian_noise_lena, gaussian_filtered_lena],
        ["Original Lena", "Gaussian noise Lena", "Gaussian filtered Lena"],
        vrange=(0, 1),
    )

    # Exploration of noisy image
    noisy_image = io.imread(NOISY_IMAGE_ASSET_PATH)
    noisy_image_spectrum = spectrum(noisy_image)
    show(
        [noisy_image, np.log1p(np.abs(noisy_image_spectrum))],
        ["Noisy image", "Noisy image spectrum"],
    )
    print(f"noisy_image shape: {noisy_image.shape}")

    # Filter design
    frequnoisy = io.imread(NOISY_IMAGE_ASSET_PATH)
    show([frequnoisy], ["Noisy image"])

    r_y = 120  # vertical radius
    r_x = 25  # horizontal radius

    rows, cols = frequnoisy.shape
    h_freq2 = np.zeros((rows, cols))

    cy2, cx2 = rows // 2, cols // 2

    Y, X = np.ogrid[-r_y : r_y + 1, -r_x : r_x + 1]

    h2 = (X / r_x) ** 2 + (Y / r_y) ** 2 <= 1

    h_freq2[cy2 - r_y : cy2 + r_y + 1, cx2 - r_x : cx2 + r_x + 1] = h2
    rotated_h_freq2 = transform.rotate(h_freq2, -20)

    show(
        [rotated_h_freq2],
        ["Filter in frequency domain"],
    )

    spectrum_frequnoisy = spectrum(frequnoisy)
    filtered_frequnoisy_spectrum = rotated_h_freq2 * spectrum_frequnoisy
    filtered_frequnoisy = np.fft.ifft2(
        np.fft.ifftshift(filtered_frequnoisy_spectrum)
    ).real

    show(
        [frequnoisy, filtered_frequnoisy],
        ["Noisy image", "Filtered image"],
    )

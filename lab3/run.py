from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color, io, transform, util
from skimage.metrics import peak_signal_noise_ratio
from skimage.morphology import disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"
NOISY_IMAGE_ASSET_PATH = ASSETS_DIR / "frequnoisy.tif"


def spectrum(image):
    return np.fft.fftshift(np.fft.fft2(image))


def show(images, titles, cmap="gray", size=4, vrange=None):
    _, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    for ax, image, title in zip(np.atleast_1d(axes), images, titles):
        if vrange is None:
            ax.imshow(image, cmap=cmap)
        else:
            ax.imshow(image, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def log_magnitude(fourier_image):
    return np.log1p(np.abs(fourier_image))


def show_denoised(original, noisy, denoised, title):
    psnr = peak_signal_noise_ratio(original, denoised, data_range=1)
    show(
        [original, noisy, denoised],
        ["Original Lena", "Noisy Lena", f"{title}\nPSNR: {psnr:.2f} dB"],
        vrange=(0, 1),
    )


def ideal_low_pass_filter(shape, radius):
    rows, cols = shape
    cy, cx = rows // 2, cols // 2
    frequency_filter = np.zeros(shape)
    frequency_filter[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1] = (
        disk(radius)
    )
    return frequency_filter


def step_1_fourier_analysis():
    f = np.zeros((256, 256))
    f[:, 107:148] = 1
    rotated_f = transform.rotate(f, 45)

    show(
        [f, np.abs(spectrum(f))],
        ["Test image", "Fourier spectrum"],
    )
    show(
        [rotated_f, np.abs(spectrum(rotated_f))],
        ["Test image rotated 45°", "Fourier spectrum of rotated image"],
    )

    lena_gray = color.rgb2gray(io.imread(LENA_ASSET_PATH))
    lena_fourier = np.fft.fft2(lena_gray)
    magnitude = np.abs(lena_fourier)
    phase = lena_fourier / magnitude

    reconstructed_amplitude = np.fft.ifft2(magnitude)
    reconstructed_phase = np.fft.ifft2(phase)
    show(
        [
            lena_gray,
            np.log1p(np.abs(reconstructed_amplitude)),
            np.abs(reconstructed_phase),
        ],
        [
            "Original (grayscale)",
            "Amplitude reconstruction (log magnitude)",
            "Phase reconstructed",
        ],
    )


def step_2_noise_reduction():
    lena_gray = color.rgb2gray(io.imread(LENA_ASSET_PATH))
    noisy_lena = util.random_noise(lena_gray, mode="gaussian", mean=0, var=0.005)
    original_log_spectrum = log_magnitude(spectrum(lena_gray))
    noisy_spectrum = spectrum(noisy_lena)
    noisy_log_spectrum = log_magnitude(noisy_spectrum)

    show(
        [original_log_spectrum, noisy_log_spectrum],
        ["Original Lena log-magnitude spectrum", "Noisy Lena log-magnitude spectrum"],
        vrange=(0, max(original_log_spectrum.max(), noisy_log_spectrum.max())),
    )

    for radius in (60, 20):
        frequency_filter = ideal_low_pass_filter(lena_gray.shape, radius)
        if radius == 60:
            show(
                [frequency_filter],
                ["Ideal low-pass filter (radius=60)"],
                vrange=(0, 1),
            )

        filtered_spectrum = frequency_filter * noisy_spectrum
        filtered_lena = np.clip(
            np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real, 0, 1
        )
        show_denoised(
            lena_gray, noisy_lena, filtered_lena, f"Ideal low-pass (radius={radius})"
        )

    rows, cols = lena_gray.shape
    gaussian_filter_mask = np.zeros_like(lena_gray)
    gaussian_filter_mask[rows // 2, cols // 2] = 1
    gaussian_filter_mask = gaussian_filter(gaussian_filter_mask, sigma=60, mode="wrap")
    gaussian_filter_mask /= gaussian_filter_mask.max()

    filtered_spectrum = gaussian_filter_mask * noisy_spectrum
    filtered_lena = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real

    show([gaussian_filter_mask], ["Gaussian frequency filter (sigma=60)"])
    show_denoised(lena_gray, noisy_lena, filtered_lena, "Gaussian low-pass (sigma=60)")


def step_3_filter_design():
    noisy_image = io.imread(NOISY_IMAGE_ASSET_PATH)
    noisy_spectrum = spectrum(noisy_image)
    show(
        [noisy_image, log_magnitude(noisy_spectrum)],
        ["Noisy image", "Noisy image log-magnitude spectrum"],
    )

    radius_y, radius_x = 120, 25
    rows, cols = noisy_image.shape
    cy, cx = rows // 2, cols // 2
    frequency_filter = np.zeros((rows, cols))
    y, x = np.ogrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    ellipse = (x / radius_x) ** 2 + (y / radius_y) ** 2 <= 1
    frequency_filter[
        cy - radius_y : cy + radius_y + 1,
        cx - radius_x : cx + radius_x + 1,
    ] = ellipse
    frequency_filter = transform.rotate(frequency_filter, -20)

    show([frequency_filter], ["Filter in frequency domain"])

    filtered_spectrum = frequency_filter * noisy_spectrum
    show(
        [log_magnitude(filtered_spectrum)],
        ["Filtered log-magnitude spectrum"],
    )
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real
    show(
        [noisy_image, filtered_image],
        ["Noisy image", "Filtered image"],
    )


def run_lab_3():
    step_1_fourier_analysis()
    step_2_noise_reduction()
    step_3_filter_design()

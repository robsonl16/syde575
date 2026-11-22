from pathlib import Path
from skimage import io, transform, util
import numpy as np
from skimage.morphology import disk
import matplotlib.pyplot as plt
from skimage import restoration
from selector_ex import Selector
from scipy import signal
from utils.plotting import save_image_grid
from utils.psnr import PSNR
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
DEGRADED_ASSET_PATH = ASSETS_DIR / "degraded.tif"


def run_lab_4():
    cameraman_gray = io.imread(CAMERAMAN_ASSET_PATH, as_gray=True)
    f = cameraman_gray / 255.0
    h_d = disk(4)
    h = np.zeros((256, 256))
    h[0:9, 0:9] = h_d

    h = np.roll(h, (-4, -4), axis=(0, 1)) / np.sum(h)
    h_freq = np.fft.fft2(h)
    f_blurfreq = h_freq * np.fft.fft2(f)
    f_blur = np.real(np.fft.ifft2(f_blurfreq))
    blurred_psnr = PSNR(f, f_blur, max_pixel_value=1.0)

    save_image_grid(
        [f, f_blur],
        ["Original", f"Disk blurred\nPSNR: {blurred_psnr:.2f} dB"],
        columns=2,
    )

    restored_freq = f_blurfreq / h_freq
    restored = np.real(np.fft.ifft2(restored_freq))
    restored_psnr = PSNR(f, restored, max_pixel_value=1.0)

    save_image_grid(
        [f, restored],
        ["Original", f"Inverse filter restored \nPSNR: {restored_psnr:.2f} dB"],
        columns=2,
    )

    noisy_spatial = util.random_noise(f_blur, mode="gaussian", mean=0, var=0.002)
    f_blur_noisy_freq = np.fft.fft2(noisy_spatial)
    restored_noisy_freq = f_blur_noisy_freq / h_freq
    restored_noisy = np.real(np.fft.ifft2(restored_noisy_freq))
    restored_noisy_psnr = PSNR(f, restored_noisy, max_pixel_value=1.0)

    save_image_grid(
        [f, restored_noisy],
        ["Original", f"Noisy and restored \nPSNR: {restored_noisy_psnr:.2f} dB"],
        columns=2,
    )

    h_centered = np.fft.fftshift(h)
    plt.imshow(h_centered, cmap="gray")
    plt.title("Centered PSF")
    plt.show()
    nsr_1 = 0.002 / np.var(f)
    nsr_2 = nsr_1 * 10
    nsr_3 = nsr_1 / 10
    f_blur_noisy = np.fft.ifft2(f_blur_noisy_freq).real
    restored_wiener_1 = restoration.wiener(f_blur_noisy, h_centered, nsr_1)
    restored_wiener_2 = restoration.wiener(f_blur_noisy, h_centered, nsr_2)
    restored_wiener_3 = restoration.wiener(f_blur_noisy, h_centered, nsr_3)
    wiener_1_psnr = PSNR(f, restored_wiener_1, max_pixel_value=1.0)
    wiener_2_psnr = PSNR(f, restored_wiener_2, max_pixel_value=1.0)
    wiener_3_psnr = PSNR(f, restored_wiener_3, max_pixel_value=1.0)

    save_image_grid(
        [f, restored_wiener_1, restored_wiener_2, restored_wiener_3],
        [
            "Original",
            f"Wiener Filter (NSR=0.002/var(f)) \nPSNR: {wiener_1_psnr:.2f} dB",
            f"Wiener Filter (NSR=Original*10) \nPSNR: {wiener_2_psnr:.2f} dB",
            f"Wiener Filter (NSR=Original/10) \nPSNR: {wiener_3_psnr:.2f} dB",
        ],
        columns=4,
    )

    degraded_gray = io.imread(DEGRADED_ASSET_PATH, as_gray=True)
    degraded_gray = degraded_gray / 255.0

    ax = plt.gca()
    ax.imshow(degraded_gray, cmap="gray")
    # Create Selector class with the current plot axis
    select = Selector(ax)
    # Show the result
    plt.show()

    # After drawing the box, retrieve the bounding box
    # This is in [row_low, row_high, col_low, col_high] format.
    flat_pixels = degraded_gray[
        select.bbox[0] : select.bbox[1], select.bbox[2] : select.bbox[3]
    ]

    print(np.size(flat_pixels))
    sigma_n_squared = np.var(flat_pixels)

    mn = np.ones((5, 5)) / 25
    local_mean = signal.convolve(degraded_gray, mn, mode="same")
    local_var = signal.convolve(degraded_gray**2, mn, mode="same") - local_mean**2

    alpha = (local_var - sigma_n_squared) / local_var

    restored_image = alpha * degraded_gray + (1 - alpha) * local_mean
    restored_image_psnr = PSNR(f, restored_image, max_pixel_value=1.0)
    save_image_grid(
        [degraded_gray, restored_image],
        [
            "Original Degraded Image",
            f"Adaptive Wiener Filter \nPSNR: {restored_image_psnr:.2f} dB",
        ],
        columns=2,
    )

    rows, cols = degraded_gray.shape
    gaussian_mask = np.zeros_like(degraded_gray)
    gaussian_mask[rows // 2, cols // 2] = 1
    gaussian_mask = gaussian_filter(gaussian_mask, sigma=30, mode="wrap")
    gaussian_mask /= gaussian_mask.max()

    degraded_spectrum = np.fft.fftshift(np.fft.fft2(degraded_gray))
    gaussian_restored = np.fft.ifft2(
        np.fft.ifftshift(gaussian_mask * degraded_spectrum)
    ).real
    gaussian_psnr = PSNR(f, gaussian_restored, max_pixel_value=1.0)

    save_image_grid(
        [degraded_gray, restored_image, gaussian_restored],
        [
            "Original Degraded Image",
            f"Adaptive Wiener Filter \nPSNR: {restored_image_psnr:.2f} dB",
            f"Gaussian Filter \nPSNR: {gaussian_psnr:.2f} dB",
        ],
        columns=3,
    )

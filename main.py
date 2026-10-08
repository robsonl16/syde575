from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import color, exposure, io, transform, util

from lab1.run import run, show_upsampling_comparison, show_image_and_histogram
from utils.psnr import PSNR

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "lab1" / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"


def imnoise_speckle(im, v):
    # im: input image
    # v: variance
    n = np.sqrt(v * 12) * (np.random.rand(im.shape[0], im.shape[1]) - 0.5)
    return im + im * n


if __name__ == "__main__":
    # run()

    # Part 1
    lena = io.imread(LENA_ASSET_PATH)
    cameraman = io.imread(CAMERAMAN_ASSET_PATH)
    cameraman_grey = util.img_as_float(cameraman)  # uint8 -> floats in [0, 1]

    lena_gray = color.rgb2gray(np.array(lena))

    # h1 = (1 / 6) * np.ones((1, 6))
    # h2 = h1.T
    # h3 = np.array([[-1, 1]])

    # lena_h1 = signal.convolve2d(lena_gray, h1, mode="same", boundary="symm")
    # lena_h2 = signal.convolve2d(lena_gray, h2, mode="same", boundary="symm")
    # lena_h3 = signal.convolve2d(lena_gray, h3, mode="same", boundary="symm")

    # show_upsampling_comparison(
    #     original_image=lena_gray,
    #     upsampled_images=[lena_h1, lena_h2, lena_h3],
    #     image_name="Lena",
    #     psnr_values=[0, 0, 0],
    # )

    # Part 2
    # f = np.hstack([0.3 * np.ones((200, 100)), 0.7 * np.ones((200, 100))])
    # gaussian_noise = util.random_noise(f, mode="gaussian", mean=0, var=0.01)
    # salt_pepper_noise = util.random_noise(f, mode="s&p", amount=0.05)
    # speckle_noise = imnoise_speckle(f, 0.04)

    # show_image_and_histogram(f, title="Original Image", intensity_range=(0, 1))
    # show_image_and_histogram(
    #     gaussian_noise, title="Gaussian Noise", intensity_range=(0, 1)
    # )
    # show_image_and_histogram(
    #     salt_pepper_noise, title="Salt and Pepper Noise", intensity_range=(0, 1)
    # )
    # show_image_and_histogram(
    #     speckle_noise, title="Speckle Noise", intensity_range=(0, 1)
    # )

    # Part 3
    gaussian_noise_p3 = util.random_noise(lena_gray, mode="gaussian", mean=0, var=0.002)
    lena_psnr = PSNR(lena_gray, gaussian_noise_p3)
    show_image_and_histogram(lena_gray, title="Original Image", intensity_range=(0, 1))
    show_image_and_histogram(
        gaussian_noise_p3,
        title=f"Gaussian Noise (PSNR: {lena_psnr:.2f})",
        intensity_range=(0, 1),
    )
    kernel = np.ones((3, 3)) / (3.0 * 3.0)
    avg_filtered = ndimage.convolve(gaussian_noise_p3, kernel)
    filtered_psnr = PSNR(lena_gray, avg_filtered)
    show_image_and_histogram(
        avg_filtered,
        title=f"Average Filtered Image (PSNR: {filtered_psnr:.2f})",
        intensity_range=(0, 1),
    )
    bigger_kernel = np.ones((7, 7)) / (7.0 * 7.0)
    more_avg_filtered = ndimage.convolve(gaussian_noise_p3, bigger_kernel)
    filtered_psnr = PSNR(lena_gray, more_avg_filtered)
    show_image_and_histogram(
        more_avg_filtered,
        title=f"Average Filtered Image (PSNR: {filtered_psnr:.2f})",
        intensity_range=(0, 1),
    )
    size = 7
    sigma = 1

    ax = np.arange(size) - (size - 1) / 2  # [-3, -2, -1, 0, 1, 2, 3]
    xx, yy = np.meshgrid(ax, ax)

    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()  # normalize so it sums to 1

    np.set_printoptions(precision=5, suppress=True)
    size = 7
    sigma = 1

    ax = np.arange(size) - (size - 1) / 2  # [-3, -2, -1, 0, 1, 2, 3]
    xx, yy = np.meshgrid(ax, ax)

    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()  # normalize so it sums to 1

    np.set_printoptions(precision=5, suppress=True)
    gaus = ndimage.convolve(gaussian_noise_p3, k)
    gaus_psnr = PSNR(lena_gray, gaus)
    show_image_and_histogram(
        gaus,
        title=f"Gaussian Filtered Image (PSNR: {gaus_psnr:.2f})",
        intensity_range=(0, 1),
    )

    salt_pepper_lena = util.random_noise(lena_gray, mode="s&p", amount=0.05)
    show_image_and_histogram(
        salt_pepper_lena, title="Salt and Pepper Noise", intensity_range=(0, 1)
    )

    salt_gaus = ndimage.convolve(gaussian_noise_p3, k)
    salt_gaus_psnr = PSNR(salt_pepper_lena, salt_gaus)
    show_image_and_histogram(
        salt_gaus,
        title=f"Salt Gaus Filtered Image (PSNR: {salt_gaus_psnr:.2f})",
        intensity_range=(0, 1),
    )
    # med_kernel = ndimage.median_filter(
    #     input,
    #     size=3,
    #     footprint=None,
    #     output=None,
    #     mode="reflect",
    #     cval=0.0,
    #     origin=0,
    #     axes=None,
    # )
    # med = ndimage.convolve(gaussian_noise_p3, med_kernel)
    # med_lena_psnr = PSNR(lena_gray, med)
    # show_image_and_histogram(
    #     med,
    #     title=f"Median Filter (PSNR: {med_lena_psnr:.2f})",
    #     intensity_range=(0, 1),
    # )

    med = ndimage.median_filter(gaussian_noise_p3, size=3)
    med_lena_psnr = PSNR(lena_gray, med)
    show_image_and_histogram(
        med,
        title=f"Median Filter (PSNR: {med_lena_psnr:.2f})",
        intensity_range=(0, 1),
    )

    sharp = ndimage.convolve(cameraman_grey, k)
    show_image_and_histogram(
        sharp,
        title="Sharpened Image)",
        intensity_range=(0, 1),
    )
    gaus_sub = cameraman_grey - sharp
    show_image_and_histogram(
        gaus_sub,
        title="Gaussian Subtracted Image)",
        intensity_range=(0, 1),
    )

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import color, exposure, io, transform, util

from utils.plotting import (
    show_filter_kernels,
    show_image_and_histogram,
    show_upsampling_comparison,
)
from utils.psnr import PSNR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"
TIRE_ASSET_PATH = ASSETS_DIR / "tire.tif"


def imnoise_speckle(im, v):
    # im: input image
    # v: variance
    n = np.sqrt(v * 12) * (np.random.rand(im.shape[0], im.shape[1]) - 0.5)
    return im + im * n


def step_1_discrete_convolution():

    lena = io.imread(LENA_ASSET_PATH)

    lena_gray = color.rgb2gray(np.array(lena))

    h1 = (1 / 6) * np.ones((1, 6))
    h2 = h1.T
    h3 = np.array([[-1, 1]])

    lena_h1 = signal.convolve2d(lena_gray, h1, mode="same", boundary="symm")
    lena_h2 = signal.convolve2d(lena_gray, h2, mode="same", boundary="symm")
    lena_h3 = signal.convolve2d(lena_gray, h3, mode="same", boundary="symm")

    show_upsampling_comparison(
        original_image=lena_gray,
        upsampled_images=[lena_h1, lena_h2, lena_h3],
        image_name="Lena",
        psnr_values=[0, 0, 0],
        intensity_range=(0, 1),
    )


def step_2_noise_generation():

    f = np.hstack([0.3 * np.ones((200, 100)), 0.7 * np.ones((200, 100))])
    gaussian_noise = util.random_noise(f, mode="gaussian", mean=0, var=0.01)
    salt_pepper_noise = util.random_noise(f, mode="s&p", amount=0.05)
    speckle_noise = imnoise_speckle(f, 0.04)

    show_image_and_histogram(f, title="Original Image", intensity_range=(0, 1))
    show_image_and_histogram(
        gaussian_noise, title="Gaussian Noise", intensity_range=(0, 1)
    )
    show_image_and_histogram(
        salt_pepper_noise, title="Salt and Pepper Noise", intensity_range=(0, 1)
    )
    show_image_and_histogram(
        speckle_noise, title="Speckle Noise", intensity_range=(0, 1)
    )


def step_3_spatial_filters():

    lena = io.imread(LENA_ASSET_PATH)
    lena_gray = color.rgb2gray(np.array(lena))

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
    gaussian_kernel = np.zeros((7, 7))
    gaussian_kernel[3, 3] = 1.0
    gaussian_kernel = ndimage.gaussian_filter(gaussian_kernel, sigma=1.0, radius=3)
    show_filter_kernels(
        {
            "7×7 Gaussian kernel (σ=1)": gaussian_kernel,
        },
        output_path=PROJECT_ROOT
        / "lab2"
        / "assets"
        / "7_by_7_gaussian_filter_kernel.png",
    )
    more_avg_filtered = ndimage.convolve(gaussian_noise_p3, bigger_kernel)
    filtered_psnr = PSNR(lena_gray, more_avg_filtered)
    show_image_and_histogram(
        more_avg_filtered,
        title=f"Average Filtered Image (PSNR: {filtered_psnr:.2f})",
        intensity_range=(0, 1),
    )

    gaussian_filtered = ndimage.gaussian_filter(gaussian_noise_p3, sigma=1.0, radius=3)
    gaussian_filtered_psnr = PSNR(lena_gray, gaussian_filtered)
    show_image_and_histogram(
        gaussian_filtered,
        title=f"Gaussian Filtered Image (PSNR: {gaussian_filtered_psnr:.2f})",
        intensity_range=(0, 1),
    )

    lena_salt_pepper_noise = util.random_noise(lena_gray, mode="s&p", amount=0.05)
    show_image_and_histogram(
        lena_salt_pepper_noise,
        title="Salt and Pepper Noise",
        intensity_range=(0, 1),
    )
    average_filtered_salt_pepper = ndimage.convolve(
        lena_salt_pepper_noise, bigger_kernel
    )
    salt_pepper_filtered_psnr = PSNR(lena_gray, average_filtered_salt_pepper)
    show_image_and_histogram(
        average_filtered_salt_pepper,
        title=f"Average Filtered Salt and Pepper Noise (PSNR: {salt_pepper_filtered_psnr:.2f})",
        intensity_range=(0, 1),
    )
    gaussian_filtered_salt_pepper = ndimage.gaussian_filter(
        lena_salt_pepper_noise, sigma=1.0, radius=3
    )
    gaussian_filtered_salt_pepper_psnr = PSNR(lena_gray, gaussian_filtered_salt_pepper)
    show_image_and_histogram(
        gaussian_filtered_salt_pepper,
        title=f"Gaussian Filtered Salt and Pepper Noise (PSNR: {gaussian_filtered_salt_pepper_psnr:.2f})",
        intensity_range=(0, 1),
    )
    median_filtered_salt_pepper = ndimage.median_filter(lena_salt_pepper_noise, size=3)
    median_filtered_salt_pepper_psnr = PSNR(lena_gray, median_filtered_salt_pepper)
    show_image_and_histogram(
        median_filtered_salt_pepper,
        title=f"Median Filtered Salt and Pepper Noise (PSNR: {median_filtered_salt_pepper_psnr:.2f})",
        intensity_range=(0, 1),
    )


def step_4_spatial_sharpening():

    cameraman = io.imread(CAMERAMAN_ASSET_PATH)
    cameraman_grey = np.array(cameraman, dtype=np.float64) / 255.0
    gaussian_filtered_cameraman = ndimage.gaussian_filter(
        cameraman_grey, sigma=1.0, radius=3
    )
    subtracted_cameraman = cameraman_grey - gaussian_filtered_cameraman
    show_image_and_histogram(
        gaussian_filtered_cameraman,
        title="Gaussian Filtered Cameraman Image",
        intensity_range=(0, 1),
    )
    show_image_and_histogram(
        subtracted_cameraman,
        title="Subtracted Cameraman Image",
        intensity_range=(0, 1),
    )

    add_subtracted_cameraman = cameraman_grey + subtracted_cameraman
    show_image_and_histogram(
        add_subtracted_cameraman,
        title="Added Subtracted Cameraman Image",
        intensity_range=(0, 1),
    )
    add_half_subtracted_cameraman = cameraman_grey + 0.5 * subtracted_cameraman
    show_image_and_histogram(
        add_half_subtracted_cameraman,
        title="Added Half Subtracted Cameraman Image",
        intensity_range=(0, 1),
    )


def run_lab_2():
    step_1_discrete_convolution()
    step_2_noise_generation()
    step_3_spatial_filters()
    step_4_spatial_sharpening()

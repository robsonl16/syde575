from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, exposure, io, transform

from utils.psnr import PSNR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "lab1" / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"
TIRE_ASSET_PATH = ASSETS_DIR / "tire.tif"


def show_upsampling_comparison(
    original_image: np.ndarray,
    upsampled_images: list[np.ndarray],
    image_name: str,
    psnr_values: list[float],
) -> None:
    methods = ["Original", "Nearest Neighbor", "Bilinear", "Bicubic"]
    figure, axes = plt.subplots(1, len(methods), figsize=(14, 4), constrained_layout=True)

    images = [original_image, *upsampled_images]
    for column, (image, method) in enumerate(zip(images, methods)):
        axes[column].imshow(image, cmap="gray", vmin=0, vmax=255)
        title = (
            method
            if column == 0
            else f"{method}\nPSNR: {psnr_values[column - 1]:.2f} dB"
        )
        axes[column].set_title(title)
        axes[column].axis("off")

    figure.suptitle(f"{image_name}: Original and Upsampled Images", fontsize=16)
    plt.show()


def show_image_and_histogram(
    image: np.ndarray,
    title: str,
    intensity_range: tuple[float, float] | None = None,
) -> None:
    """Display a grayscale image beside its 256-bin intensity histogram."""
    if intensity_range is None:
        intensity_range = (float(image.min()), float(image.max()))

    figure, (image_axis, histogram_axis) = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True
    )
    image_axis.imshow(
        image,
        cmap="gray",
        vmin=intensity_range[0],
        vmax=intensity_range[1],
    )
    image_axis.set_title(title)
    image_axis.axis("off")

    histogram_axis.hist(image.ravel(), bins=256, range=intensity_range, color="gray")
    histogram_axis.set_title(f"{title} Histogram")
    histogram_axis.set_xlabel("Intensity")
    histogram_axis.set_ylabel("Pixel count")
    histogram_axis.set_xlim(intensity_range)
    plt.show()


def show_power_law_transformations(
    transformed_images: list[np.ndarray], gammas: list[float]
) -> None:
    """Display each power-law transformed image beside its intensity histogram."""
    figure, axes = plt.subplots(
        len(transformed_images), 2, figsize=(10, 8), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for row, (image, gamma) in enumerate(zip(transformed_images, gammas)):
        image_axis, histogram_axis = axes[row]
        image_axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        image_axis.set_title(f"Power-law transform ($\\gamma = {gamma}$)")
        image_axis.axis("off")

        histogram_axis.hist(image.ravel(), bins=256, range=(0, 1), color="gray")
        histogram_axis.set_title(f"Histogram ($\\gamma = {gamma}$)")
        histogram_axis.set_xlabel("Intensity")
        histogram_axis.set_ylabel("Pixel count")
        histogram_axis.set_xlim(0, 1)

    figure.suptitle("Power-law Transformed Tire Images", fontsize=16)
    plt.show()


def step_1_digital_zooming():

    RESOLUTION_REDUCTION_FACTOR = 0.25

    # Load images
    cameraman = io.imread(CAMERAMAN_ASSET_PATH)
    lena = io.imread(LENA_ASSET_PATH)

    # Get image dimensions
    CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT = cameraman.shape[1], cameraman.shape[0]
    LENA_WIDTH, LENA_HEIGHT = lena.shape[1], lena.shape[0]

    # Convert images to greyscale and float64 for processing
    cameraman_grey = np.array(cameraman, dtype=np.float64)
    lena_grey = color.rgb2gray(np.array(lena)) * 255

    # Downsample images
    reduced_cameraman = transform.resize(
        cameraman_grey,
        (
            int(CAMERAMAN_HEIGHT * RESOLUTION_REDUCTION_FACTOR),
            int(CAMERAMAN_WIDTH * RESOLUTION_REDUCTION_FACTOR),
        ),
        order=1,
        preserve_range=True,
    )
    reduced_lena = transform.resize(
        lena_grey,
        (
            int(LENA_HEIGHT * RESOLUTION_REDUCTION_FACTOR),
            int(LENA_WIDTH * RESOLUTION_REDUCTION_FACTOR),
        ),
        order=1,
        preserve_range=True,
    )

    # Upsample images using different interpolation methods
    upsampled_cameraman_nn = transform.resize(
        reduced_cameraman,
        (CAMERAMAN_HEIGHT, CAMERAMAN_WIDTH),
        order=0,
        preserve_range=True,
    )
    upsampled_lena_nn = transform.resize(
        reduced_lena, (LENA_HEIGHT, LENA_WIDTH), order=0, preserve_range=True
    )

    upsampled_cameraman_bilinear = transform.resize(
        reduced_cameraman,
        (CAMERAMAN_HEIGHT, CAMERAMAN_WIDTH),
        order=1,
        preserve_range=True,
    )
    upsampled_lena_bilinear = transform.resize(
        reduced_lena, (LENA_HEIGHT, LENA_WIDTH), order=1, preserve_range=True
    )

    upsampled_cameraman_bicubic = transform.resize(
        reduced_cameraman,
        (CAMERAMAN_HEIGHT, CAMERAMAN_WIDTH),
        order=3,
        preserve_range=True,
    )
    upsampled_lena_bicubic = transform.resize(
        reduced_lena, (LENA_HEIGHT, LENA_WIDTH), order=3, preserve_range=True
    )

    # Calculate PSNR values for each upsampling method
    cameraman_psnr_nn = PSNR(cameraman_grey, upsampled_cameraman_nn)
    cameraman_psnr_bilinear = PSNR(cameraman_grey, upsampled_cameraman_bilinear)
    cameraman_psnr_bicubic = PSNR(cameraman_grey, upsampled_cameraman_bicubic)
    lena_psnr_nn = PSNR(lena_grey, upsampled_lena_nn)
    lena_psnr_bilinear = PSNR(lena_grey, upsampled_lena_bilinear)
    lena_psnr_bicubic = PSNR(lena_grey, upsampled_lena_bicubic)

    show_upsampling_comparison(
        original_image=cameraman_grey,
        upsampled_images=[
            upsampled_cameraman_nn,
            upsampled_cameraman_bilinear,
            upsampled_cameraman_bicubic,
        ],
        image_name="Cameraman",
        psnr_values=[
            cameraman_psnr_nn,
            cameraman_psnr_bilinear,
            cameraman_psnr_bicubic,
        ],
    )
    show_upsampling_comparison(
        original_image=lena_grey,
        upsampled_images=[
            upsampled_lena_nn,
            upsampled_lena_bilinear,
            upsampled_lena_bicubic,
        ],
        image_name="Lena",
        psnr_values=[lena_psnr_nn, lena_psnr_bilinear, lena_psnr_bicubic],
    )


def step_2_tire_image_and_histogram():

    # Load tire image
    tire = io.imread(TIRE_ASSET_PATH)

    # Scale the tire image to the range [0, 1]
    scaled_tire = tire / 255.0

    show_image_and_histogram(
        image=scaled_tire, title="Original Tire Image", intensity_range=(0, 1)
    )


def step_3_negative_transform():

    # Load tire image
    tire = io.imread(TIRE_ASSET_PATH)

    # Scale the tire image to the range [0, 1]
    scaled_tire = tire / 255.0

    # Apply negative transformation
    negative_transformed_tire = 1 - scaled_tire

    show_image_and_histogram(
        image=negative_transformed_tire,
        title="Negative Transformed Tire Image",
        intensity_range=(0, 1),
    )


def step_4_power_law_transformations():

    # Load tire image
    tire = io.imread(TIRE_ASSET_PATH)

    # Scale the tire image to the range [0, 1]
    scaled_tire = tire / 255.0

    power_law_transformed_tire_05 = exposure.adjust_gamma(scaled_tire, gamma=0.5)
    power_law_transformed_tire_13 = exposure.adjust_gamma(scaled_tire, gamma=1.3)

    show_power_law_transformations(
        transformed_images=[
            power_law_transformed_tire_05,
            power_law_transformed_tire_13,
        ],
        gammas=[0.5, 1.3],
    )


def step_5_histogram_equalization():

    # Load tire image
    tire = io.imread(TIRE_ASSET_PATH)

    # Scale the tire image to the range [0, 1]
    scaled_tire = tire / 255.0

    # Apply histogram equalization
    equalized_tire = exposure.equalize_hist(scaled_tire)

    show_image_and_histogram(
        image=equalized_tire,
        title="Histogram Equalized Tire Image",
        intensity_range=(0, 1),
    )


def run():
    step_1_digital_zooming()
    step_2_tire_image_and_histogram()
    step_3_negative_transform()
    step_4_power_law_transformations()
    step_5_histogram_equalization()

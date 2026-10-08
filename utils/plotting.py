import matplotlib.pyplot as plt
import numpy as np


def show_image_and_histogram(
    image: np.ndarray,
    title: str,
    intensity_range: tuple[float, float] | None = None,
) -> None:
    """Display a grayscale image beside its 256-bin intensity histogram."""
    if intensity_range is None:
        intensity_range = (float(image.min()), float(image.max()))

    _figure, (image_axis, histogram_axis) = plt.subplots(
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


def show_upsampling_comparison(
    original_image: np.ndarray,
    upsampled_images: list[np.ndarray],
    image_name: str,
    psnr_values: list[float],
) -> None:
    methods = ["Original", "Nearest Neighbor", "Bilinear", "Bicubic"]
    figure, axes = plt.subplots(
        1, len(methods), figsize=(14, 4), constrained_layout=True
    )

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

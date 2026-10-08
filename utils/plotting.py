from pathlib import Path

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


def show_filter_kernels(
    kernels: dict[str, np.ndarray], output_path: Path | None = None
) -> None:
    """Display filter weights as annotated heatmaps."""
    figure, axes = plt.subplots(
        1, len(kernels), figsize=(6 * len(kernels), 5), constrained_layout=True
    )
    for axis, (title, kernel) in zip(np.atleast_1d(axes), kernels.items()):
        image = axis.imshow(kernel, cmap="Blues", vmin=0)
        center_row, center_col = np.array(kernel.shape) // 2
        axis.set_xticks(
            range(kernel.shape[1]), range(-center_col, kernel.shape[1] - center_col)
        )
        axis.set_yticks(
            range(kernel.shape[0]), range(-center_row, kernel.shape[0] - center_row)
        )
        axis.set_xlabel("Horizontal offset (pixels)")
        axis.set_ylabel("Vertical offset (pixels)")
        axis.set_title(title)
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                axis.text(
                    col, row, f"{kernel[row, col]:.3g}",
                    ha="center", va="center", fontsize=7,
                    color="white" if kernel[row, col] > kernel.max() / 2 else "black",
                )
        figure.colorbar(image, ax=axis, label="Weight", shrink=0.8)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)
    plt.show()


def show_upsampling_comparison(
    original_image: np.ndarray,
    upsampled_images: list[np.ndarray],
    image_name: str,
    psnr_values: list[float],
    intensity_range: tuple[float, float] | None = [0, 255],
) -> None:
    methods = ["Original", "Nearest Neighbor", "Bilinear", "Bicubic"]
    figure, axes = plt.subplots(
        1, len(methods), figsize=(14, 4), constrained_layout=True
    )

    images = [original_image, *upsampled_images]
    for column, (image, method) in enumerate(zip(images, methods)):
        axes[column].imshow(
            image, cmap="gray", vmin=intensity_range[0], vmax=intensity_range[1]
        )
        title = (
            method
            if column == 0
            else f"{method}\nPSNR: {psnr_values[column - 1]:.2f} dB"
        )
        axes[column].set_title(title)
        axes[column].axis("off")

    figure.suptitle(f"{image_name}: Original and Upsampled Images", fontsize=16)
    plt.show()

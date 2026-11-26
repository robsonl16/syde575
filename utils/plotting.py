from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def save_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path | None = None,
    *,
    columns: int = 3,
    figure_title: str | None = None,
    roi_panel: int | None = None,
    roi: tuple[int, int, int, int] | None = None,
    cmap: str | None = "gray",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Save comparable images on a fixed [vmin, vmax] intensity scale.

    2D images use ``cmap``; 3D (H, W, 3|4) images are shown as colour.
    """
    if len(images) != len(titles) or not images:
        raise ValueError("Provide one title for each image.")
    rows = (len(images) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), constrained_layout=True
    )
    for index, (axis, image, title) in enumerate(
        zip(np.atleast_1d(axes).flat, images, titles)
    ):
        if image.ndim == 3:
            # imshow ignores vmin/vmax for colour, so normalise to [0, 1] ourselves
            axis.imshow(np.clip((image - vmin) / (vmax - vmin), 0, 1))
        else:
            axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        if index == roi_panel and roi is not None:
            row_low, row_high, col_low, col_high = roi
            axis.add_patch(
                Rectangle(
                    (col_low, row_low),
                    col_high - col_low,
                    row_high - row_low,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )
    for axis in np.atleast_1d(axes).flat[len(images) :]:
        axis.axis("off")
    if figure_title:
        figure.suptitle(figure_title)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)

    plt.show()
    plt.close(figure)


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
                    col,
                    row,
                    f"{kernel[row, col]:.3g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if kernel[row, col] > kernel.max() / 2 else "black",
                )
        figure.colorbar(image, ax=axis, label="Weight", shrink=0.8)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)
    plt.show()


def show_convolution_comparison(
    original_image: np.ndarray,
    horizontal_average: np.ndarray,
    vertical_average: np.ndarray,
    horizontal_difference: np.ndarray,
    image_name: str,
    output_path: Path | None = None,
) -> None:
    """Compare the original image with three convolution results."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    image_panels = (
        (original_image, "Original"),
        (horizontal_average, "h1: Horizontal average (1×6)"),
        (vertical_average, "h2: Vertical average (6×1)"),
        (horizontal_difference, "h3: Horizontal edge filter"),
    )
    for axis, (image, title) in zip(axes.flat, image_panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"{image_name}: Discrete Convolution")
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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from skimage import color, io, transform

from utils.psnr import PSNR

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "lab1" / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"
TIRE_ASSET_PATH = ASSETS_DIR / "tire.tif"

# TODO make helper for greyscale images even if they are already greyscale


def show_upsampling_comparison(
    original_images: list[np.ndarray],
    upsampled_images: list[list[np.ndarray]],
    image_names: list[str],
    psnr_values: list[list[float]],
) -> None:
    methods = ["Original", "Nearest Neighbor", "Bilinear", "Bicubic"]
    figure, axes = plt.subplots(
        len(original_images), len(methods), figsize=(14, 7), constrained_layout=True
    )

    for row, (original, reconstructions, image_name, scores) in enumerate(
        zip(original_images, upsampled_images, image_names, psnr_values)
    ):
        images = [original, *reconstructions]
        for column, (image, method) in enumerate(zip(images, methods)):
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=255)
            title = (
                method
                if column == 0
                else f"{method}\nPSNR: {scores[column - 1]:.2f} dB"
            )
            axes[row, column].set_title(f"{image_name}: {title}")
            axes[row, column].axis("off")

    figure.suptitle("Original and Upsampled Images", fontsize=16)
    plt.show()


def main():

    RESOLUTION_REDUCTION_FACTOR = 0.25

    cameraman = io.imread(CAMERAMAN_ASSET_PATH)
    lena = io.imread(LENA_ASSET_PATH)

    CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT = cameraman.shape[1], cameraman.shape[0]
    LENA_WIDTH, LENA_HEIGHT = lena.shape[1], lena.shape[0]

    cameraman_grey = np.array(cameraman, dtype=np.float64)
    lena_grey = color.rgb2gray(np.array(lena)) * 255

    reduced_cameraman = transform.resize(
        cameraman_grey,
        (
            int(CAMERAMAN_HEIGHT * RESOLUTION_REDUCTION_FACTOR),
            int(CAMERAMAN_WIDTH * RESOLUTION_REDUCTION_FACTOR),
        ),
        preserve_range=True,
    )
    reduced_lena = transform.resize(
        lena_grey,
        (
            int(LENA_HEIGHT * RESOLUTION_REDUCTION_FACTOR),
            int(LENA_WIDTH * RESOLUTION_REDUCTION_FACTOR),
        ),
        preserve_range=True,
    )

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

    # cameraman = Image.open(CAMERAMAN_ASSET_PATH)
    # lena = Image.open(LENA_ASSET_PATH)

    # CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT = cameraman.size
    # LENA_WIDTH, LENA_HEIGHT = lena.size

    # Using PIL
    # cameraman_grey = cameraman.convert("L")
    # lena_grey = lena.convert("L")

    # cameraman_grey.show()
    # lena_grey.show()

    # reduced_cameraman = zoom(cameraman_grey, RESOLUTION_REDUCTION_FACTOR, order=1)
    # reduced_lena = zoom(lena_grey, RESOLUTION_REDUCTION_FACTOR, order=1)

    # Image._show(Image.fromarray(reduced_cameraman))
    # Image._show(Image.fromarray(reduced_lena))

    # upsampled_cameraman_nn = Image.fromarray(reduced_cameraman).resize(
    #     (CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT), resample=Image.Resampling.NEAREST
    # )
    # upsampled_cameraman_bilinear = Image.fromarray(reduced_cameraman).resize(
    #     (CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT), resample=Image.Resampling.BILINEAR
    # )
    # upsampled_cameraman_bicubic = Image.fromarray(reduced_cameraman).resize(
    #     (CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT), resample=Image.Resampling.BICUBIC
    # )
    # upsampled_lena_nn = Image.fromarray(reduced_lena).resize(
    #     (LENA_WIDTH, LENA_HEIGHT), resample=Image.Resampling.NEAREST
    # )
    # upsampled_lena_bilinear = Image.fromarray(reduced_lena).resize(
    #     (LENA_WIDTH, LENA_HEIGHT), resample=Image.Resampling.BILINEAR
    # )
    # upsampled_lena_bicubic = Image.fromarray(reduced_lena).resize(
    #     (LENA_WIDTH, LENA_HEIGHT), resample=Image.Resampling.BICUBIC
    # )

    cameraman_psnr_nn = PSNR(cameraman_grey, upsampled_cameraman_nn)
    cameraman_psnr_bilinear = PSNR(cameraman_grey, upsampled_cameraman_bilinear)
    cameraman_psnr_bicubic = PSNR(cameraman_grey, upsampled_cameraman_bicubic)
    lena_psnr_nn = PSNR(lena_grey, upsampled_lena_nn)
    lena_psnr_bilinear = PSNR(lena_grey, upsampled_lena_bilinear)
    lena_psnr_bicubic = PSNR(lena_grey, upsampled_lena_bicubic)

    print(
        f"PSNR between Cameraman and Cameraman (Nearest Neighbor): {cameraman_psnr_nn:.2f} dB"
    )
    print(
        f"PSNR between Cameraman and Cameraman (Bilinear): {cameraman_psnr_bilinear:.2f} dB"
    )
    print(
        f"PSNR between Cameraman and Cameraman (Bicubic): {cameraman_psnr_bicubic:.2f} dB"
    )
    print(f"PSNR between Lena and LENA (Nearest Neighbor): {lena_psnr_nn:.2f} dB")
    print(f"PSNR between Lena and LENA (Bilinear): {lena_psnr_bilinear:.2f} dB")
    print(f"PSNR between Lena and LENA (Bicubic): {lena_psnr_bicubic:.2f} dB")

    show_upsampling_comparison(
        original_images=[cameraman_grey, lena_grey],
        upsampled_images=[
            [
                upsampled_cameraman_nn,
                upsampled_cameraman_bilinear,
                upsampled_cameraman_bicubic,
            ],
            [upsampled_lena_nn, upsampled_lena_bilinear, upsampled_lena_bicubic],
        ],
        image_names=["Cameraman", "Lena"],
        psnr_values=[
            [
                cameraman_psnr_nn,
                cameraman_psnr_bilinear,
                cameraman_psnr_bicubic,
            ],
            [lena_psnr_nn, lena_psnr_bilinear, lena_psnr_bicubic],
        ],
    )


if __name__ == "__main__":
    main()

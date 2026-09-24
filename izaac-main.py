from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.io import imread
from skimage.transform import resize

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "lab1" / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"

cameraman = imread(CAMERAMAN_ASSET_PATH)
lena = imread(LENA_ASSET_PATH)

# image quality assessment


def PSNR_test(img, alt_img):

    MAX_f = 255.0

    img_f = np.array(img, dtype=np.float64)
    alt_img_f = np.array(alt_img, dtype=np.float64)

    # MSE = np.sum((img_f - alt_img_f) ** 2) / (img_f.shape[0] * img_f.shape[1])
    MSE = np.mean((img_f - alt_img_f) ** 2)

    if MSE == 0:
        return float("inf")
    PSNR = 10 * np.log10((MAX_f**2) / MSE)

    # PSNR = 10 * np.log10((MAX_f**2) / MSE)

    print("MSE:", MSE)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(alt_img, cmap="gray")
    plt.title(f"Altered\nPSNR: {PSNR:.2f} dB")
    plt.axis("off")

    return PSNR


# Digital zooming
def digital_zoom(img):

    # comment out this line for cameraman.
    # img = color.rgb2gray(img)

    new_shape = (img.shape[0] // 4, img.shape[1] // 4)
    up_size = (new_shape[0] * 4, new_shape[1] * 4)

    # nearest neighbour
    img_mean = resize(img, new_shape, order=1)
    img_mean_up = resize(img_mean, up_size, order=0)
    PSNR_NN = PSNR_test(img, img_mean_up)

    # Bilinear
    img_bilinear = resize(img, new_shape, order=1)
    img_bilinear_up = resize(img_bilinear, up_size, order=1)
    PSNR_BL = PSNR_test(img, img_bilinear_up)

    # Bicubic
    img_bicubic = resize(img, new_shape, order=1)
    img_bicubic_up = resize(img_bicubic, up_size, order=3)
    PSNR_BC = PSNR_test(img, img_bicubic_up)

    # One figure, 4 images
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img_mean_up, cmap="gray")
    plt.title(f"Nearest Neighbour\nPSNR: {PSNR_NN:.2f} dB")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img_bilinear_up, cmap="gray")
    plt.title(f"Bilinear\nPSNR: {PSNR_BL:.2f} dB")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(img_bicubic_up, cmap="gray")
    plt.title(f"Bicubic\nPSNR: {PSNR_BC:.2f} dB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


digital_zoom(cameraman)

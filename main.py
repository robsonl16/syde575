from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import color, exposure, io, transform

from lab1.run import run, show_upsampling_comparison
from utils.psnr import PSNR

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "lab1" / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"

if __name__ == "__main__":
    # run()

    # Part 1
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
    )

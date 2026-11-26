from lab1.run import run_lab_1
from lab2.run import run_lab_2
from lab3.run import run_lab_3
from lab4.run import run_lab_4

from pathlib import Path
from skimage import io, transform, util, color
import numpy as np
from skimage.morphology import disk
import matplotlib.pyplot as plt
from skimage import restoration
from selector_ex import Selector
from scipy import signal
from utils.plotting import save_image_grid
from utils.psnr import PSNR
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
DEGRADED_ASSET_PATH = ASSETS_DIR / "degraded.tif"
PEPPERS_ASSET_PATH = ASSETS_DIR / "peppers.png"


if __name__ == "__main__":
    # run_lab_1()
    # run_lab_2()
    # run_lab_3()
    # run_lab_4()

    peppers = io.imread(PEPPERS_ASSET_PATH)
    peppers_ycbcr = color.rgb2ycbcr(peppers)
    peppers_y = peppers_ycbcr[:, :, 0]
    peppers_cb = peppers_ycbcr[:, :, 1]
    peppers_cr = peppers_ycbcr[:, :, 2]

    save_image_grid(
        [peppers_y, peppers_cb, peppers_cr],
        ["Y channel", "Cb channel", "Cr channel"],
        columns=3,
        cmap=None,
        vmin=0,
        vmax=255,
    )

    PEPPERS_HEIGHT, PEPPERS_WIDTH = peppers.shape[0], peppers.shape[1]
    peppers_cb_downsampled = transform.resize(
        peppers_cb,
        (PEPPERS_HEIGHT // 2, PEPPERS_WIDTH // 2),
        order=1,
        preserve_range=True,
    )
    peppers_cr_downsampled = transform.resize(
        peppers_cr,
        (PEPPERS_HEIGHT // 2, PEPPERS_WIDTH // 2),
        order=1,
        preserve_range=True,
    )

    peppers_cb_upsampled = transform.resize(
        peppers_cb_downsampled,
        (PEPPERS_HEIGHT, PEPPERS_WIDTH),
        order=1,
        preserve_range=True,
    )
    peppers_cr_upsampled = transform.resize(
        peppers_cr_downsampled,
        (PEPPERS_HEIGHT, PEPPERS_WIDTH),
        order=1,
        preserve_range=True,
    )

    peppers_2_ycbcr = np.stack(
        (peppers_y, peppers_cb_upsampled, peppers_cr_upsampled), axis=-1
    )
    peppers_2_rgb = color.ycbcr2rgb(peppers_2_ycbcr) * 255.0

    save_image_grid(
        [peppers, peppers_2_rgb],
        ["Original", "Downsampled and upsampled"],
        columns=2,
        cmap=None,
        vmin=0,
        vmax=255,
    )

    peppers_y_downsampled = transform.resize(
        peppers_y,
        (PEPPERS_HEIGHT // 2, PEPPERS_WIDTH // 2),
        order=1,
        preserve_range=True,
    )
    peppers_y_upsampled = transform.resize(
        peppers_y_downsampled,
        (PEPPERS_HEIGHT, PEPPERS_WIDTH),
        order=1,
        preserve_range=True,
    )
    peppers_3_ycbcr = np.stack((peppers_y_upsampled, peppers_cb, peppers_cr), axis=-1)
    peppers_3_rgb = color.ycbcr2rgb(peppers_3_ycbcr) * 255.0
    save_image_grid(
        [peppers, peppers_3_rgb],
        ["Original", "Downsampled and upsampled Y channel"],
        columns=2,
        cmap=None,
        vmin=0,
        vmax=255,
    )

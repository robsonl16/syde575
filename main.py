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
from sklearn import cluster

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

    peppers_lab = color.rgb2lab(peppers)

    # K = 2
    # row = np.array([55, 200]) - 1
    # col = np.array([155, 400]) - 1

    K = 4
    row = np.array([55, 130, 200, 280]) - 1
    col = np.array([155, 110, 400, 470]) - 1

    mu = peppers_lab[row, col]

    m, n, ch = peppers_lab.shape
    peppers_lab_reshaped = np.reshape(peppers_lab, (m * n, ch), order="F")

    cluster_idx = cluster.KMeans(n_clusters=K, init=mu, n_init=1).fit_predict(
        peppers_lab_reshaped
    )

    pixel_labels = np.reshape(cluster_idx, (m, n), order="F")
    cluster_1 = np.zeros((m, n, ch))
    cluster_2 = np.zeros((m, n, ch))
    cluster_3 = np.zeros((m, n, ch))
    cluster_4 = np.zeros((m, n, ch))
    for i in range(m):
        for j in range(n):
            if pixel_labels[i, j] == 0:
                cluster_1[i, j, :] = 1
            elif pixel_labels[i, j] == 1:
                cluster_2[i, j, :] = 1
            elif pixel_labels[i, j] == 2:
                cluster_3[i, j, :] = 1
            elif pixel_labels[i, j] == 3:
                cluster_4[i, j, :] = 1

    cluster_1 = np.multiply(peppers_lab, cluster_1)
    cluster_2 = np.multiply(peppers_lab, cluster_2)
    cluster_3 = np.multiply(peppers_lab, cluster_3)
    cluster_4 = np.multiply(peppers_lab, cluster_4)
    save_image_grid(
        [
            color.lab2rgb(cluster_1),
            color.lab2rgb(cluster_2),
            color.lab2rgb(cluster_3),
            color.lab2rgb(cluster_4),
        ],
        [
            "Cluster 1",
            "Cluster 2",
            "Cluster 3",
            "Cluster 4",
        ],
        columns=4,
    )

from lab1.run import run_lab_1
from lab2.run import run_lab_2
from lab3.run import run_lab_3

from pathlib import Path
from skimage import io, transform, util
import numpy as np
from skimage.morphology import disk
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"

if __name__ == "__main__":
    # run_lab_1()
    # run_lab_2()
    # run_lab_3()
    cameraman_gray = io.imread(CAMERAMAN_ASSET_PATH, as_gray=True)
    f = cameraman_gray / 255.0
    h_d = disk(4)
    h = np.zeros((256, 256))
    h[0:9, 0:9] = h_d

    h = np.roll(h, (-5, -5)) / np.sum(h)
    h_freq = np.fft.fft2(h)
    f_blurfreq = h_freq * np.fft.fft2(f)
    f_blur = np.real(np.fft.ifft2(f_blurfreq))

    restored_freq = f_blurfreq / h_freq
    restored = np.real(np.fft.ifft2(restored_freq))

    plt.imshow(f_blur, cmap="gray")
    plt.title("Blurred Image")
    plt.show()
    plt.imshow(restored, cmap="gray")
    plt.title("Restored Image")
    plt.show()

    noisy = util.random_noise(f, mode="gaussian", mean=0, var=0.002)
    noisy_freq = np.fft.fft2(noisy)
    blurred_noisy_freq = h_freq * noisy_freq
    restored_noisy_freq = blurred_noisy_freq / h_freq
    restored_noisy = np.real(np.fft.ifft2(restored_noisy_freq))
    plt.imshow(restored_noisy, cmap="gray")
    plt.title("Restored Noisy Image")
    plt.show()

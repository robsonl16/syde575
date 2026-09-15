import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from skimage import color

from utils.psnr import PSNR

CAMERAMAN_ASSET_PATH = (
    "C:\\Users\\robso_jdkjtyb\\Documents\\GitHub\\syde575\\lab1\\assets\\cameraman.tif"
)
LENA_ASSET_PATH = (
    "C:\\Users\\robso_jdkjtyb\\Documents\\GitHub\\syde575\\lab1\\assets\\lena.tif"
)
TIRE_ASSET_PATH = (
    "C:\\Users\\robso_jdkjtyb\\Documents\\GitHub\\syde575\\lab1\\assets\\tire.tif"
)

# TODO make helper for greyscale images even if they are already greyscale


def main():
    cameraman = Image.open(CAMERAMAN_ASSET_PATH)
    lena = Image.open(LENA_ASSET_PATH)

    CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT = cameraman.size
    LENA_WIDTH, LENA_HEIGHT = lena.size

    cameraman_grey = cameraman.convert("L")
    lena_grey = lena.convert("L")

    cameraman_grey.show()
    lena_grey.show()

    RESOLUTION_REDUCTION_FACTOR = 0.25

    reduced_cameraman = zoom(cameraman_grey, RESOLUTION_REDUCTION_FACTOR, order=1)
    reduced_lena = zoom(lena_grey, RESOLUTION_REDUCTION_FACTOR, order=1)

    Image._show(Image.fromarray(reduced_cameraman))
    Image._show(Image.fromarray(reduced_lena))

    upsampled_cameraman = Image.fromarray(reduced_cameraman).resize(
        (CAMERAMAN_WIDTH, CAMERAMAN_HEIGHT), resample=Image.Resampling.NEAREST
    )
    upsampled_lena = Image.fromarray(reduced_lena).resize(
        (LENA_WIDTH, LENA_HEIGHT), resample=Image.Resampling.NEAREST
    )

    cameraman_psnr = PSNR(cameraman_grey, upsampled_cameraman)
    lena_psnr = PSNR(lena_grey, upsampled_lena)
    print(f"PSNR between Cameraman and Cameraman: {cameraman_psnr:.2f} dB")
    print(f"PSNR between Lena and LENA: {lena_psnr:.2f} dB")


if __name__ == "__main__":
    main()

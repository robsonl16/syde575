import numpy as np
from PIL import Image


def PSNR(
    reference_img: Image.Image | np.ndarray, test_img: Image.Image | np.ndarray
) -> float:

    if reference_img.size != test_img.size:
        raise ValueError("Input images must have the same dimensions.")

    MAX_PIXEL_VALUE = 255.0

    # Convert images to numpy arrays of type float64
    reference_array = np.array(reference_img, dtype=np.float64)
    test_array = np.array(test_img, dtype=np.float64)

    # Calculate MSE
    mse = np.mean((reference_array - test_array) ** 2)

    # Calculate PSNR
    if mse == 0:
        return float("inf")
    psnr = 10 * np.log10((MAX_PIXEL_VALUE**2) / mse)

    return psnr

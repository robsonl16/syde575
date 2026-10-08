from pathlib import Path

from lab1.run import run_lab_1
from lab2.run import run_lab_2

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

CAMERAMAN_ASSET_PATH = ASSETS_DIR / "cameraman.tif"
LENA_ASSET_PATH = ASSETS_DIR / "lena.tif"


if __name__ == "__main__":
    # run_lab_1()
    run_lab_2()

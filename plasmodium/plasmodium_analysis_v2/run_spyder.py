"""Open this file in Spyder (lbr environment) and press Run."""

from multiprocessing import freeze_support

from plasmodium_v2 import run_analysis
from user_config import SETTINGS


if __name__ == "__main__":
    freeze_support()
    RUN_OUTPUT_DIRECTORY = run_analysis(SETTINGS)


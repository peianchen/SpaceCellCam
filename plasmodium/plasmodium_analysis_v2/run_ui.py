"""Open in Spyder (lbr) and press Run to launch the desktop UI."""
from multiprocessing import freeze_support
from plasmodium_v2.ui_app import launch_ui
from user_config import SETTINGS

if __name__ == "__main__":
    freeze_support()
    UI_WINDOW = launch_ui(SETTINGS)

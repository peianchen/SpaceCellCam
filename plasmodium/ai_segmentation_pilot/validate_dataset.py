from pathlib import Path

from plasmodium_ai import validate_dataset


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(validate_dataset(root / "pilot_config.json"))


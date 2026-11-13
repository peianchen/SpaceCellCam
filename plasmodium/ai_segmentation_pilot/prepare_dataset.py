from pathlib import Path

from plasmodium_ai import prepare_dataset


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    output = prepare_dataset(root / "pilot_config.json")
    print(f"DATASET_READY={output}")


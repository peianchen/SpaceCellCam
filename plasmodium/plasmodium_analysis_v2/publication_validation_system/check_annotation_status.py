"""Report missing primary, repeat, adjudicated, and prediction masks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from common import VALID_ROLES, load_json, package_root


def rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_masks(directory: Path, expected_ids: list[str]):
    present = []
    invalid = []
    for mask_id in expected_ids:
        path = directory / f"{mask_id}.png"
        if not path.exists():
            continue
        present.append(mask_id)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or int(image.max()) == 0:
            invalid.append(mask_id)
    missing = sorted(set(expected_ids) - set(present))
    return len(present), missing, invalid


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument("--role", choices=sorted(VALID_ROLES), required=True)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    root = package_root(config, args.config.resolve(), args.role)
    manifest = rows(root / "private_manifest.csv")
    if not manifest:
        raise SystemExit(f"No package manifest found under {root}")
    primary_ids = [row["blind_id"] for row in manifest]
    repeat_manifest = rows(root / "private_repeat_manifest.csv")
    repeat_ids = [row["repeat_blind_id"] for row in repeat_manifest]
    checks = []
    for annotator in config["annotation"]["annotators"]:
        checks.append((annotator, root / "annotators" / annotator / "masks", primary_ids))
    for annotator in config["annotation"].get("repeat_annotators", []):
        checks.append((f"{annotator}_repeat", root / "annotators" / f"{annotator}_repeat" / "masks", repeat_ids))
    checks.append(("adjudicated", root / "adjudicated_masks", primary_ids))
    checks.extend((f"prediction_{method}", root / "predictions" / method, primary_ids) for method in ("strict", "temporal"))
    for name, directory, expected in checks:
        present, missing, invalid = count_masks(directory, expected)
        print(f"{name}: {present}/{len(expected)} present, {len(missing)} missing, {len(invalid)} empty/unreadable")
        if missing:
            print("  missing sample:", ", ".join(missing[:5]))
        if invalid:
            print("  empty/unreadable sample:", ", ".join(invalid[:5]))


if __name__ == "__main__":
    main()

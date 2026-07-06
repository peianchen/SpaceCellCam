"""Freeze the analysis source and segmentation parameters for final validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import (
    algorithm_parameter_payload,
    canonical_sha256,
    load_json,
    resolve_from,
    sha256_file,
    write_json,
)


def freeze(config_path: Path) -> Path:
    config = load_json(config_path)
    algorithm = config["algorithm"]
    source_path = resolve_from(config_path, algorithm["source_file"])
    runner_path = resolve_from(config_path, algorithm["prediction_runner_file"])
    if not source_path.exists():
        raise FileNotFoundError(f"Algorithm source does not exist: {source_path}")
    if not runner_path.exists():
        raise FileNotFoundError(f"Prediction runner does not exist: {runner_path}")

    source_hash = sha256_file(source_path)
    runner_hash = sha256_file(runner_path)
    parameter_hash = canonical_sha256(algorithm_parameter_payload(config))
    frozen_at = datetime.now(timezone.utc).isoformat()
    algorithm["status"] = "frozen"
    algorithm["locked_source_sha256"] = source_hash
    algorithm["locked_prediction_runner_sha256"] = runner_hash
    algorithm["locked_parameters_sha256"] = parameter_hash
    algorithm["frozen_at_utc"] = frozen_at
    write_json(config_path, config)

    lock_path = config_path.parent / "algorithm_lock.json"
    write_json(
        lock_path,
        {
            "frozen_at_utc": frozen_at,
            "source_file": str(source_path),
            "source_sha256": source_hash,
            "prediction_runner_file": str(runner_path),
            "prediction_runner_sha256": runner_hash,
            "parameters_sha256": parameter_hash,
            "parameters": algorithm_parameter_payload(config),
            "rule": (
                "Any source or parameter change invalidates this lock. If test results "
                "have been viewed, a new independent test set is required after refitting."
            ),
        },
    )
    return lock_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=script_dir / "config.json")
    parser.add_argument(
        "--confirm-development-complete",
        action="store_true",
        help="Required acknowledgement that no further parameter tuning is planned.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_development_complete:
        raise SystemExit(
            "Refusing to freeze without --confirm-development-complete. "
            "Freeze only after all development and validation-set tuning is finished."
        )
    lock_path = freeze(args.config.resolve())
    print(f"Algorithm frozen: {lock_path}")


if __name__ == "__main__":
    main()

"""Per-run output isolation, metadata, and artifact inventory."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import AnalysisSettings
from .performance import PerformancePlan


def _safe_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_")


def create_output_directory(settings: AnalysisSettings) -> Path:
    if settings.resume_run_directory is not None:
        return Path(settings.resume_run_directory).expanduser().resolve()
    root = Path(settings.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not settings.unique_run_directory:
        return root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _safe_label(settings.run_label)
    name = f"{stamp}_{settings.run_mode}" + (f"_{suffix}" if suffix else "")
    candidate = root / name
    counter = 2
    while candidate.exists():
        candidate = root / f"{name}_{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_initial_metadata(
    settings: AnalysisSettings, plan: PerformancePlan, output_dir: Path, engine_file: Path
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 2,
        "status": "running",
        "started_at_utc": now,
        "run_mode": settings.run_mode,
        "output_directory": str(output_dir),
        "settings": settings.to_dict(),
        "performance": plan.__dict__,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "engine_sha256": sha256_file(engine_file),
    }
    path = output_dir / "run_manifest.json"
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"run_manifest_{settings.run_mode}_{stamp}.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        [{
            "run_mode": settings.run_mode,
            "status": "running",
            "started_at_utc": now,
            "output_directory": str(output_dir),
            "cpu_workers": plan.workers,
            "parallel_batch_frames": plan.batch_frames,
        }]
    ).to_csv(output_dir / "run_mode_indicator.csv", index=False, encoding="utf-8-sig")
    return path


def finalize_metadata(manifest_path: Path, status: str, error: str | None = None) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["status"] = status
    data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error:
        data["error"] = error
    output_dir = manifest_path.parent
    data["artifacts"] = sorted(
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
    )
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


"""Stable Spyder-friendly entrypoint that keeps worker functions importable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import AnalysisSettings
from .performance import build_performance_plan
from .run_context import create_output_directory, finalize_metadata, write_initial_metadata


RunEventHook = Callable[[str, dict[str, Any]], None]


def _notify(hook: RunEventHook | None, event: str, **payload: Any) -> None:
    if hook is not None:
        hook(event, payload)


def run_analysis(
    settings: AnalysisSettings,
    event_hook: RunEventHook | None = None,
) -> Path:
    settings.validate()
    plan = build_performance_plan(settings)
    output_dir = create_output_directory(settings)
    runtime = settings.with_runtime(
        output_dir=output_dir,
        cpu_workers=plan.workers,
        batch_frames=plan.batch_frames,
    )
    from . import engine

    manifest = write_initial_metadata(runtime, plan, output_dir, Path(engine.__file__))
    _notify(
        event_hook,
        "output_created",
        output_directory=str(output_dir),
        manifest_path=str(manifest),
    )
    print(f"New isolated run directory: {output_dir}")
    print(
        f"Performance plan: {plan.workers}/{plan.cpu_count} workers, "
        f"batch={plan.batch_frames}, memory budget={plan.memory_budget_gb:.1f} GiB"
    )
    try:
        engine.run(runtime)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            finalize_metadata(manifest, "failed", f"SystemExit({exc.code})")
            _notify(event_hook, "failed", error=f"SystemExit({exc.code})")
            raise
    except BaseException as exc:
        finalize_metadata(manifest, "failed", f"{type(exc).__name__}: {exc}")
        _notify(event_hook, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finalize_metadata(manifest, "completed")
    _notify(event_hook, "completed", output_directory=str(output_dir))
    print(f"Run completed: {output_dir}")
    return output_dir


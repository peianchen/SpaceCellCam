"""Spawn-safe worker used by the desktop UI."""
from __future__ import annotations
import contextlib
import os
import traceback
import psutil
from .config import AnalysisSettings
from .runner import run_analysis

class QueueWriter:
    def __init__(self, queue, stream):
        self.queue, self.stream = queue, stream
    def write(self, text):
        if text:
            self.queue.put({"type": "log", "stream": self.stream, "text": text})
        return len(text)
    def flush(self):
        pass

def _cleanup_analysis_children() -> None:
    """Ensure failed or cancelled nested pools do not remain as RAM-heavy orphans."""
    try:
        children = psutil.Process().children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(children, timeout=3)
        for child in alive:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def run_analysis_worker(settings: AnalysisSettings, queue) -> None:
    os.environ["PLASMODIUM_UI_WORKER"] = "1"
    def event_hook(event, payload):
        queue.put({"type": "event", "event": event, **payload})
    try:
        with contextlib.redirect_stdout(QueueWriter(queue, "stdout")), contextlib.redirect_stderr(QueueWriter(queue, "stderr")):
            output_dir = run_analysis(settings, event_hook=event_hook)
        queue.put({"type": "done", "output_directory": str(output_dir)})
    except BaseException as exc:
        queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        _cleanup_analysis_children()

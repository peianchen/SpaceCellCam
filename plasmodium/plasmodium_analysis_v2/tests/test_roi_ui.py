import builtins

import numpy as np

from plasmodium_v2 import engine


def test_roi_selection_uses_default_name_when_worker_has_no_stdin(monkeypatch):
    monkeypatch.setattr(engine.cv2, "selectROIs", lambda *args, **kwargs: np.array([[4, 5, 20, 15]]))
    monkeypatch.setattr(engine.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(engine.cv2, "waitKey", lambda _delay: 0)

    def no_stdin(*_args, **_kwargs):
        raise EOFError("worker has no stdin")

    monkeypatch.setattr(builtins, "input", no_stdin)
    rois = engine.select_contraction_rois_ui(np.zeros((64, 64, 3), dtype=np.uint8))
    assert rois == [{"name": "roi_0", "x": 4, "y": 5, "w": 20, "h": 15}]

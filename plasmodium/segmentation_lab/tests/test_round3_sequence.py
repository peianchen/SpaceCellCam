"""Exercise the real sequence IO contract using a tiny synthetic video."""
import json
import cv2
import numpy as np
import pytest
from segmentation_lab.round3_sequence import run_sequence


def test_sequence_respects_explicit_fixed_roi_and_marks_untrusted_pair(tmp_path):
    video=tmp_path/'blank.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5.,(128,128))
    assert writer.isOpened()
    for _ in range(2):writer.write(np.full((128,128,3),85,np.uint8))
    writer.release()
    config=tmp_path/'roi.json'
    config.write_text(json.dumps({'roi':{'center':[64,64],'radius':48,'margin_px':3},
        'new_method_roi_by_video':{'example':{}},
        'methods':{'adaptive_chroma':{'refine_roi':False,'food_exclusion':False}}}),encoding='utf-8')
    out=tmp_path/'run'
    result=run_sequence(video,out,roi_config=config,video_id='example',count=2)
    assert result['status']=='complete'
    assert result['base_config']['refine_roi'] is False
    assert result['biological_time_known'] is False
    qc=json.loads((out/'frame_000001/qc.json').read_text(encoding='utf-8'))
    assert qc['segmentation']['roi_refined'] is False
    assert not qc['pair']['diagnostics']['valid']
    assert all(value is None for value in qc['pair']['overlap'].values())
    mask=cv2.imread(str(out/'frame_000001/mask.png'),0)
    assert not mask.any()
    with pytest.raises(FileExistsError):
        run_sequence(video,out,roi_config=config,video_id='example',count=2)

import cv2
import numpy as np

from plasmodium_v2.manual_validation import export_manual_validation_package


def test_manual_export_uses_temporal_masks_and_skips_edges(tmp_path):
    video = tmp_path / "manual.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 96), True
    )
    assert writer.isOpened()
    for index in range(15):
        frame = np.full((96, 96, 3), 60, np.uint8)
        cv2.circle(frame, (45 + index // 5, 48), 20, (15, 180, 205), -1)
        writer.write(frame)
    writer.release()
    hsv = dict(
        h_min=15, h_max=49, s_min=54, v_min=30, min_obj_size=50,
        keep_largest=True, use_yellow_chroma_extension=False,
        use_vesselness_recovery=False,
    )
    temporal = dict(
        window=5, min_hits=4, chroma_min=10, min_channel=35, max_value=240,
        rim_exclusion_px=5, connect_radius=4, min_obj_size=5,
        exterior_only=True, block_hole_max_area=0,
        final_smooth_window=1, final_smooth_min_hits=1,
    )
    manifest = export_manual_validation_package(
        video, tmp_path / "out", (48, 48), 44, hsv,
        samples_per_section=1, section_frames=3, edge_skip=2,
        use_temporal_recovery=True, temporal_kwargs=temporal,
    )
    assert not manifest.empty
    assert set(manifest["auto_mask_method"]) == {"temporal_recovered_spatial_mask"}
    assert manifest["frame"].min() >= 2
    assert manifest["frame"].max() <= 12
    assert manifest["frame"].is_unique


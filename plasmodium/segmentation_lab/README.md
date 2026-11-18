# 黏菌分割與震動診斷：第二輪候選版

更新：2026-11-18。這是可獨立執行、可替換的分割與檢查工具。版本摘要與已完成的測試見 [第二輪里程碑](milestones/20261118-round2.md)。V2 仍是原分析系統的操作入口；本工具不自動接管正式速度、面積、骨架或收縮分析。

本次主要處理暗光細網漏抓與 RPM 旋轉造成的影像位移。使用者確認這批影像品質較差，邊緣另有對焦模糊，之後會有品質較好的影像。這個版本保留為新影像到來前的比較基準；新資料仍需實測，不能先假設結果會全面改善。

## 遮罩定義

白色前景代表影像可見的黏菌覆蓋。排除可辨識的食物本體，保留可見的食物表面黏菌；網絡中的背景孔洞不填滿。無法從單張影像判定被食物遮住的組織或過度失焦的細網。

`occupied` 為最終組織 mask，`clean=occupied`；`envelope_area_px` 是填洞後的參考範圍，不能用作組織面積或骨架。

## 本版本做了什麼

- `segmenter.py`：第一輪主體／食物／ROI 分割，保留為可重現基準。
- `segmenter_round2.py`：局部雜訊與多尺度線條特徵補回弱分支；不做時間填補或平滑。
- `registration_v2.py`：外側候選固定特徵的平移與旋轉估計，使用未參與擬合的點檢查，證據不足就拒絕。
- `round2_sequence.py`：串接逐張分割與相鄰影格診斷，輸出原圖、mask、overlay、QC、變換與來源紀錄。
- `annotations.py`／`mask_editor.py`：草稿、人工確認及評分分開；沒有確認的真值時準確率為空值。
- `benchmark.py`／`round2_benchmark.py`／`verify_round2.py`：保留舊方法、並排比較、來源核對與最終程式重播。

全部使用傳統影像特徵；沒有訓練新的 AI 模型，也沒有將 SAM2 或弱標註當成人工真值。

## 執行新影片

從儲存庫的 `plasmodium/` 目錄執行。使用已有科學影像套件的 Python 環境；本次測試使用 Python 3.12.12，套件版本見 [environment_tested.json](environment_tested.json)。它是版本紀錄，不是新環境安裝鎖檔；原 V2 的依賴起點見 [requirements-lbr.txt](../plasmodium_analysis_v2/requirements-lbr.txt)。

先將 [roi_config.example.json](roi_config.example.json) 複製為 `roi_config.local.json`，按照原圖修改培養區中心、半徑及影片識別碼。範例數字不是實驗校正值。輸出目錄每次必須使用新名稱。

```powershell
python -m segmentation_lab.round2_sequence --video 'D:\experiment\video.mp4' --start 0 --count 8 --step 1 --roi-config segmentation_lab/roi_config.local.json --video-id example_video --output segmentation_lab/outputs/check_001
python -m pytest segmentation_lab/tests -q
```

每張輸出都在原圖座標。`registration_valid=False` 的影格對不提供校正後量測；返回原位矩陣不表示成功消震。即使外側特徵對齊，`biological_plane_correction_verified` 仍為 `False`，因為外側標記與培養底面可能有不同表觀運動。

播放 FPS 不等於真實取像間隔。只有實驗紀錄確定每張原圖的間隔後，才加入 `--acquisition-interval-sec`；未確認前不報真實速度。

## 單張介面

```python
from segmentation_lab.contracts import RoiSpec
from segmentation_lab.segmenter_round2 import segment

result = segment(bgr_uint8_image, RoiSpec((1280, 960), 650, 10),
                 base_config={"refine_roi": True})  # 幾何僅為範例
occupied = result.mask
```

`score` 是特徵證據，不是正確機率。保留 `roi_mask`、`excluded_food`、`qc_flags` 和 `diagnostics` 一起檢查。使用快取基準時須經 `prepare_baseline` 綁定原圖、ROI 與設定，不能跨影格混用。

## 結果在哪裡

Git 保存程式、通用參數、測試、設定範例與 [里程碑摘要](milestones/20261118-round2.md)。實驗影片、逐影格清單、實際 ROI 設定、遮罩、疊圖、詳細運行紀錄及快照留在本機，遵循 [資料管理規則](../docs/REPOSITORY_POLICY.md)。

在實際執行本輪的工作站，可開啟以下本機產物；重新 clone 不會包含它們：

- `segmentation_lab/ROUND2_RESULTS.html`：中文比較入口。
- `segmentation_lab/outputs/round2_comparison32/gallery.html`：32 張原圖／前後疊圖／新增像素。
- `segmentation_lab/outputs/round2_motion_geometry_corrected/`：四段影片的位移診斷、疊圖及 GIF。
- `segmentation_lab/outputs/round2_final_replay/verification.json`：32 張重播核對。
- `segmentation_lab/README_ROUND1_LOCAL.md`：保留的第一輪完整本機操作記錄。

完整基準比較需要操作者自己的原始資料與 manifest。不能把缺少本機資料的 clone 當成已重跑真實影像；合成測試可以直接執行。

## 已知限制與下一輪

暗處的部分枝段有補回，但極暗／失焦細網仍斷裂，反光與孤立淡斑仍需覆核。影像對齊可補償部分幾何位移，無法恢復失焦細節或已形成的運動模糊。新法為加法，也不會主動移除第一輪誤抓。

收到較清楚的影像後，先用這個固定版本與參數輸出原圖／mask／overlay；記錄曝光、焦距、真實拍攝間隔與 ROI，再決定是否調參。優先補上人工確認的細網與反光案例，並以與培養底面同平面的固定標記驗證震動補償。缺少真值時，不報 Dice／Recall 改善。

方法與資料定義另見 [原程式查核](BASELINE_AUDIT.md)、[定量查核](QUANTIFICATION_AUDIT.md)、[人工標註規則](ANNOTATION_PROTOCOL.md)。

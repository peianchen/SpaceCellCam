# 第三輪：暗光細網候選與震動可靠度檢查

本版保留 R1、R2，新增可獨立執行的 R3。它仍是待人工覆核的分割候選，沒有訓練新 AI，也沒有把弱標註或 SAM2 輸出當成人工真值。正式 V2 分析入口不變。

## 改了什麼

暗線即使肉眼可沿著方向辨認，逐像素門檻仍可能把它切成碎段。`segmenter_round3.py` 使用不同粗細、方向的線狀濾波器，要求線中心的黃色訊號比左右兩側都強，再搭配局部對比與連通證據。沿線整合訊號可以保留部分弱分支；沒有當下影像證據的空隙不以時間平滑或填洞補上。

R1 的主體、食物排除及 ROI 原樣保留。R2 新增的小型孤立圓斑，以及 R3 在高亮外緣提出的新候選，會依保守規則保留為待確認區。這些區域可能是真實短枝或小團塊，不能直接稱為錯誤；此規則也無法修正 R1 已有的誤抓。`withheld_review.png` 保存受影響範圍供人工查看。

`registration_v3.py` 沿用 R2 的剛體位移估計，新增兩類接受檢查：修正量要能與剩餘不一致區分，且有足夠特徵點的區域不能明顯變差。它減少勉強套用校正的情況；通過時矩陣與 R2 相同，不能把拒絕數增加解讀成消震效果增加。外圈對齊仍不代表黏菌所在平面已對齊。

## 執行

從 `plasmodium/` 目錄執行；環境與幾何設定準備見 [原操作說明](README.md)。每次使用新輸出目錄。參數單位為原圖像素，不保證適用其他拍攝倍率。

```powershell
python -m segmentation_lab.round3_sequence --video 'D:\experiment\video.mp4' --start 0 --count 8 --step 1 --roi-config segmentation_lab/roi_config.local.json --video-id example_video --output segmentation_lab/outputs/r3_check_001
python -m pytest segmentation_lab/tests -q
```

連續流程逐張輸出原圖、二值 mask、疊圖、ROI 及 QC，保存原始像素 hash、實際解碼影格、配準矩陣及相鄰影格診斷。下方的固定清單比較工具另輸出食物排除、前處理顯示、特徵強度與待確認範圍。逐張 mask 不因前後影格而修改。配準不可信時，校正後的探索／退縮診斷為空，不能把回傳的單位矩陣當成零震動。未提供真實取像間隔時，不填實驗時間或速度。

單張使用：

```python
from segmentation_lab.contracts import RoiSpec
from segmentation_lab.segmenter_round3 import segment

debug = {}
result = segment(bgr_uint8_image, RoiSpec((1280, 960), 650, 10),
                 base_config={"refine_roi": True}, debug=debug)
occupied = result.mask
```

幾何數字僅為範例。`score` 及 `quality_score` 是啟發式證據，不是機率或準確率。`occupied=clean`，填洞的 `envelope_area_px` 只供參考，不用來計組織面積。`debug` 可取得方向性證據及裁切座標，包含 R2 結果供同圖比較。

固定清單比較需要操作者自己的原始 PNG、pixel hash、影片路徑及幾何設定：

```powershell
python -m segmentation_lab.round3_benchmark --manifests path/to/old_results.json path/to/new_samples.json --geometry segmentation_lab/roi_config.local.json --config segmentation_lab/config_round3.json --output segmentation_lab/outputs/r3_comparison_001
```

舊清單有 `round2_mask_path` 時會驗證 R2 重播一致。每張保留原圖／R2／R3／新增移除位置。橘色與藍色是變更位置，不是已知正確或錯誤。食物邊界紫色，組織邊界綠色。

## 驗證與限制

本輪實際檢查範圍及版本來源見 [第三輪里程碑](milestones/20261118-round3.md)。完整研究影像、逐影格結果與人工覆核記錄留在本機 `outputs/`，不隨公開程式庫推送。

最暗、失焦和反光交疊的細網仍不可靠。反光保留規則依初始圓的外側範圍判定，並未涵蓋精修橢圓的每一處邊界；最終組織 mask 仍受精修 ROI 限制。R3 可能讓枝條變粗，也可能排除真正的孤立短枝；無人工真值不能算真實 IoU、Dice、Precision 或 Recall。更嚴格的配準檢查也可能拒絕有用的小位移。下一步應以人工確認的暗網／反光遮罩，以及培養平面固定標記的對位資料，驗證是否值得接入正式生物定量。單靠數值更平順不構成驗證。

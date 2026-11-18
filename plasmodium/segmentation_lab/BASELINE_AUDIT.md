# 現有程式 baseline 查核（2026-11-18）

本次 wrapper 使用現有來源產生共同原圖解析度的**單張空間分割**。沒有執行舊程式主流程，也不重跑完整影片。它保留舊方法的食物誤抓與固定閾值限制，方便公平顯示差異；不應把較大的 mask 自動解讀為較準確。

## 呼叫與來源

`run_baseline(image_bgr, roi: RoiSpec, method: str) -> SegmentationResult` 支援 `legacy_original`、`v2_spatial`。影像必須是原圖座標的 `uint8` BGR；不額外白平衡、縮放或做時間補回。

- `legacy_original`：安全擷取原版 `if __name__ == "__main__"` 內的 `hsv_params`，呼叫原版 `segment_hsv_circle`。來源：[原版設定](../plasmodium_brightfield_analysis.py#L6018)、[原版函式](../plasmodium_brightfield_analysis.py#L1151)。
- `v2_spatial`：只讀取 `engine.run` 內的 `hsv_params`，實際呼叫 V2 `segmentation.py`；不執行 `engine.run`。來源：[V2 設定](../plasmodium_analysis_v2/plasmodium_v2/engine.py#L8408)、[V2 函式](../plasmodium_analysis_v2/plasmodium_v2/segmentation.py#L110)。
- wrapper 只接受 literal `dict(...)` 設定，透過 AST 解析，沒有執行設定運算。每次結果記錄實際讀到的來源 SHA-256、完整參數與設定行號；實作以該來源快照載入，`__name__` 不會是 `__main__`。V2 只載入 segmentation 與 mask_utils，避免啟動 UI 或 orchestration。
- 2026-11-18 已用現有 lbr Python、`-B`、`MPLBACKEND=Agg` 實際確認原版、optimized 與 V2 都能 import；沒有執行 main/run。

## 版本差異

| 項目 | 原版 | optimized | V2 |
|---|---|---|---|
| 主入口 HSV | H15–49、S≥54、V≥30 | 相同 | 相同 |
| 單張黃色延伸 | 關閉 | 開啟，chroma≥23 | 相同 |
| 附近 vesselness 補回 | 無 | 開啟，名義 2% 上限 | 開啟，嚴格 2% 上限 |
| 時間候選投票 | 5 張中 3 張 | 5 張中 4 張 | 5 張中 4 張 |
| 最後時間平滑 | 5 張中 3 張，可移除當張像素 | 預設關閉 | 預設關閉，實作保留當張空間 mask |
| 時間平移對齊 | 無 | 無 | 有，僅用於時間候選投票 |
| 正式運算解析度 | 0.5 | 1.0 | 1.0 |

設定來源：原版 5936–5961、6018–6053；optimized 8819–8846、8964–9011；V2 engine 8254–8290、8408–8455。**函式預設參數不能當主程式設定**：函式預設 S=60，但入口使用 S=54。

optimized 的 vesselness cap 在首個連通區大於上限時仍可整區加入（1462–1481）；V2 在 segmentation.py 86–106 改為不能超過上限。故 optimized 與 V2 即使參數一樣，結果也不保證完全相同。

原版目前預設 `RUN_MODE="test_front_middle_rear"`（5858）。它的 validation `output_scale=0.5` 會在分割前縮圖（5093–5109），不只是縮小輸出影片。wrapper 刻意統一原解析度，結果**不等於舊預設完整重播**。原版完整時間流程還有 5+5 的組合時間窗，單張結果不能代表它；若只重播中央影格，需要足夠前後文，而 V2 還須固定 batch/anchor 才能談逐像素一致。

## 載入、ROI 與 mask 定義

- 原版載圖 `load_frames_from_folder(folder, as_gray=True)`、載片 `load_frames_from_video(path, max_frames=None, as_gray=True)` 分別在 124、162 行；主入口在 6192 行對 HSV 設定 `as_gray=False`。比較器也必須保持 BGR：灰階轉回三色仍沒有飽和度，會使黃色分割失敗。
- 圖片資料夾採檔名文字排序（137 行），不保證數字自然排序；讀取失敗會略過。影片載入 `read()` 失敗即結束（188–190），單靠成功載入不能證明完整解碼。
- 原版/optimized 圓 ROI 預設中心 `(1280,920)`、半徑 `680`；V2 設定亦相同。這不是自動偵測培養皿。不同尺寸／拍攝位置必須提供對應座標。矩形 `ANALYSIS_ROI` 只控制 classic/grabcut 路徑，不能拿它當 HSV 的裁切範圍（原版 5900–5903、1844–1857）。
- wrapper 接受比較器提供的同一 ROI，忠實使用舊演算法整數中心／半徑與 **margin=0**；`RoiSpec.margin_px` 不額外侵蝕 baseline ROI，requested/effective geometry 都記錄。若外部重估了 ROI，這屬於共同測試條件，不等於原本 hardcoded geometry 的重播。
- `occupied_mask` 是實際篩出的可見像素；`clean_mask` 是 occupied 加上可選的小孔填補；`envelope_mask` 是把 enclosed holes 全填掉的覆蓋參考，**不是可見黏菌面積**。三版主入口 `fill_small_body_holes=False`，因此目前 spatial clean=occupied。原版定義見 1308–1377；optimized 1687–1766；V2 segmentation 222–288。
- 原版一般量測由 `_body_mask_layers` 取 clean mask 計面積/骨架/前後變化（1863–1891）；時間流程則直接將 temporal mask 同時作 occupied/clean，再另算 envelope（2674–2706）；V2 相同（engine 3117–3181）。wrapper 回報 occupied，並在 diagnostics 分開記錄三層面積與 clean 的新增／移除像素。

## 解讀限制與失敗來源

固定 HSV／色差受照明、白平衡與反光影響；opening=1、小物件刪除150、keep_largest=True 會遺失斷開細枝；黃色色差須接到現有種子，沒有種子不能自動找回網路。原版半解析度還會使細管與弱色訊號混入背景。三版舊方法都沒有語意區分食物，棕黃色食物可能被列入 occupied。這些風險由來源可確認，實際改善幅度須看保留的真實影像與人工參考，不能由 mask 面積或程式測試代替。

baseline `score` 只是 0/1 occupied mask，沒有機率意義；`excluded_food` 全 false，QC 包含 `baseline_food_not_excluded`。`preprocessed` 是原 BGR 副本。`candidate` 是舊 debug 候選聯集，其中部分已受最大連通區過濾，不能稱為完整未篩選候選。

`body_mask_confidence` 本身只是洞面積比例的啟發式（原版 1123–1126、V2 engine 1221–1224），並非人工標註準確率；真實網狀黏菌有很多洞，也可能被標 low，因此比較器不能以該欄當改善評分。

## 本次測試

現有 lbr Python 執行 `python -B -m pytest segmentation_lab/tests/test_baselines.py -q -p no:cacheprovider`：**7 passed，6.02 s**。涵蓋 AST 不執行設定、主入口 S=54、保留原圖、洞與 ROI 語義、V2 黃色延伸、來源紀錄可 JSON 序列化、空白／錯誤輸入。這些是合成小圖與 wrapper 行為測試；沒有評估真實黏菌標註準確率。

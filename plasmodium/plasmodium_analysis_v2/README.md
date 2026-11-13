# Plasmodium Analysis V2

這是從舊單檔程式獨立建立的新專案。舊的
`plasmodium_brightfield_analysis_optimized.py` 與舊結果資料夾不會被修改。

照片合併、照片轉影片、變速、壓縮與 GIF 工具已整理於
[`../video_tools/README.md`](../video_tools/README.md)。它們獨立執行，產生的影片可再由分析介面選取；
播放 FPS 與真實取像間隔須分別設定，變速展示副本不可直接當成原始時間序列。

## PyQt 桌面介面（建議）

新下載專案時，先在本目錄把 `user_config.example.py` 複製成 `user_config.py`，
再填入實際輸入與校正值；已有本機設定時請保留。`run_ui.py` 和 `run_spyder.py` 都需要此檔。
完整操作順序見 [目前流程](../docs/CURRENT_WORKFLOW.md)，後續修改見 [開發指南](../docs/DEVELOPMENT.md)。

1. 確認 Spyder interpreter 是 `C:\Users\chen8\.conda\envs\lbr\python.exe`。
2. 在 Spyder 開啟 `run_ui.py`，直接按 Run。
3. 在「基本設定」選擇影片、輸出位置與執行模式。
4. 按「開始分析」，程式會切到執行日誌頁；分析在獨立程序執行，UI 不會因長時間計算停止更新。

介面提供輸入/輸出瀏覽、全部 run mode、培養皿設定、CPU/RAM 預估、驗證設定、即時日誌、停止分析與開啟結果資料夾。 PyQt 背景程序沒有互動式終端，因此 OpenCV 選取 ROI 後會自動命名為 `roi_0`、`roi_1`…。按「儲存介面設定」會寫入本機 `ui_settings.json`，該檔已排除於 Git。若從 UI 停止，當次 manifest 會記錄為 `cancelled`。

## 在 Spyder 直接執行設定檔

如果不需要 UI，仍可沿用原本方式：

1. 確認 Spyder interpreter 是：

   ```text
   C:\Users\chen8\.conda\envs\lbr\python.exe
   ```

2. 開啟 `user_config.py`，修改輸入影片、模式、培養皿位置與輸出設定。
3. 開啟 `run_spyder.py`，直接按 Run。

請不要直接執行 `plasmodium_v2/engine.py`。由 `run_spyder.py` 啟動時，worker
函式屬於可匯入的 package module，Windows/Spyder multiprocessing 會比較穩定。

## 模式切換

在 `user_config.py` 修改：

```python
run_mode="validation_all"
```

可用模式：

- `formal_only`：全影片正式分析。
- `formal_with_validation`：正式分析加三段驗證。
- `validation_all`：前、中、後段全解析度 temporal 驗證。
- `validation_front`、`validation_middle`、`validation_rear`：單區段驗證。
- `contraction_roi`：ROI contraction/tracker。
- `manual_validation_export`：輸出與正式 temporal 方法一致的人工標註套件。
- `manual_validation_annotate`、`manual_validation_score`：標註與評分。
- `complete`：執行全部流程。

## 新版輸出管理

每次執行會建立獨立資料夾，例如：

```text
runs/20260714_165128_validation_front/
```

每個資料夾都有：

- `run_manifest.json`：狀態、完整設定、Python、OpenCV、CPU/記憶體規劃、engine hash、產物清單。
- `run_mode_indicator.csv`：本次模式與實際遮罩方法。
- 本次模式產生的 CSV、圖、影片與驗證檔。

因此不會再把本次 `validation_all` 和舊的 `formal_only` CSV 混在同一層。

人工驗證是三步連續工作。第一次用 `manual_validation_export` 建立新 run；後兩步
必須在 `user_config.py` 指定同一個資料夾：

```python
run_mode="manual_validation_annotate"
resume_run_directory=Path(r"...\runs\20260714_170000_manual_validation_export")
```

評分時把模式改成 `manual_validation_score`，保留相同的
`resume_run_directory`。每個續接動作會留下自己的 manifest，不覆蓋最初 export
的 manifest。

## 多核心與記憶體

預設：

```python
cpu_workers=0
parallel_batch_frames=0
memory_fraction=0.55
```

`0` 表示自動配置。程式會：

- 讀取實際 CPU 數量及可用 RAM。
- 預留一個 CPU core 給 Spyder/Windows。
- 本機最多使用 15 個 worker（受 `max_cpu_workers=16` 限制）。
- 依可用記憶體選擇 batch 大小。
- 在 worker 內關閉 OpenCV 額外執行緒，避免程序與執行緒重複超載。
- 在正式 streaming 分析中重用同一個 process pool，不再每個 batch 重開。
- `contraction_roi` 使用 temporal 分批串流與懶載入 tracker，不再把三段 300 張全解析度 BGR 影格同時放進 RAM。
- contraction 預設一次處理一個 early/middle/late 段、段內多核心；UI 停止或分析失敗時會清理巢狀 worker，避免殘留程序持續占用記憶體。

如果 Spyder 或記憶體不穩，可在 `user_config.py` 明確設定：

```python
cpu_workers=8
parallel_batch_frames=16
memory_fraction=0.40
```

## 程式架構

```text
run_ui.py                    PyQt5 桌面介面入口
run_spyder.py                 無 UI 的 Spyder 執行入口
user_config.py                日常修改設定的位置
plasmodium_v2/config.py       型別化設定與檢查
plasmodium_v2/performance.py  CPU/RAM 自動調校
plasmodium_v2/run_context.py  獨立輸出與 manifest
plasmodium_v2/runner.py       執行生命週期、事件與失敗紀錄
plasmodium_v2/ui_app.py       PyQt5 視窗、表單、日誌與程序控制
plasmodium_v2/ui_worker.py    背景分析程序與日誌轉送
plasmodium_v2/mask_utils.py   共用二值遮罩工具
plasmodium_v2/segmentation.py HSV/chroma/vesselness
plasmodium_v2/temporal.py     temporal recovery 與 smoothing
plasmodium_v2/manual_validation.py  temporal 人工驗證
plasmodium_v2/validation_api.py     論文驗證使用的最小穩定 API
plasmodium_v2/engine.py       既有形態、動態、ROI、繪圖相容引擎
publication_validation_system/     V2 獨立盲測、凍結與評估流程
```

`engine.py` 仍保留舊程式中已驗證的形態、contraction、tracker、UI 與繪圖函式，
但日常會調整的分割、temporal、設定、效能和輸出生命週期已拆成獨立模組。

## 已修正項目

- 每次執行獨立輸出，避免舊結果混用。
- contraction-only 也會在開始前寫入 manifest/indicator。
- validation indicator 正確標記 temporal mask。
- `strict_hsv` 改以 `spatial_non_temporal` 語意記錄，同時保留舊欄位相容性。
- 正式 CSV 加入 yellow-chroma 與 vesselness 貢獻欄位。
- vesselness added fraction 為真正硬上限。
- 所有 validation 模式使用原始解析度分割，只縮小輸出影片。
- manual validation 評分正式 temporal mask，並排除 temporal 邊緣影格。
- temporal smoothing 不會移除當前 spatial mask。
- temporal batch halo 支援兩階段 smoothing。
- Temporal voting 會先用培養皿邊緣估算平移並對齊 mask；低可信度或超過 30 px
  的估計會自動退回原座標，避免轉動影片的逐幀震動把細管共識抹掉。
- 影片中途解碼失敗會明確報錯，不會輸出缺幀但看似成功的 CSV。
- 單區段驗證不會刪除其他區段資料夾。

## 測試

在 Anaconda Prompt：

```powershell
conda activate lbr
cd C:\Users\chen8\Documents\PlatformIO\Projects\SpaceCellCam\plasmodium\plasmodium_analysis_v2
python -m pytest -q tests
python -m tests.run_end_to_end_smoke
python publication_validation_system/synthetic_smoke_test.py
```

V2 publication workflow 會把 `validation_api.py`、`segmentation.py`、
`temporal.py` 與 `mask_utils.py` 合併計算 source hash；其中任一模組改變都會使正式
lock 失效。預設仍是 `development_not_frozen`，synthetic smoke 只證明軟體接線，
不代表生物影像準確度。


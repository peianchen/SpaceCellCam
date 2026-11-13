# 目前實驗影像處理流程

更新：2026-11-14。本頁說明現有程式的實際入口和使用順序；物種、培養條件、設備校正與生物效應仍須由實驗紀錄確認。檔案或專案名稱中的 *Physarum* 不構成物種鑑定。

## 先選對入口

| 工作 | 目前入口 | 定位 |
|---|---|---|
| 相機取像與 SD 寫入 | 儲存庫根目錄的 `src/`、`platformio.ini`、[根 README](../../README.md) | 嵌入式取像韌體 |
| 照片整理及影片轉換 | [video_tools](../video_tools/README.md) | 獨立前後處理工具 |
| 日常影像分析 | `plasmodium_analysis_v2/run_ui.py` | 建議主入口；選檔、設定、執行及檢查日誌 |
| 設定檔執行 | `plasmodium_analysis_v2/run_spyder.py` | 讀取 `user_config.py`，適合 Spyder 或終端 |
| 獨立盲測／方法凍結 | `plasmodium_analysis_v2/publication_validation_system/` | V2 專用驗證流程，現有缺口見下文 |
| AI 標註可行性探索 | [ai_segmentation_pilot](../ai_segmentation_pilot/README.md) | 獨立支線，尚未接入 V2 正式分析 |
| 歷史重現或比較 | `plasmodium_brightfield_analysis_*.py`、`plasmodium_sample_*.py` | 保留舊版；不要因檔名有 optimized/formal 就當成目前主入口 |

## 1. 取像與原始資料保存

依根 README 和實際韌體設定記錄相機、LED、對焦、取像間隔與 SD 檔案結構。現行設計使用 `/bXXX/image_<n>.jpg` 分桶；設定的 10 秒週期不代表每張照片都已證實沒有延遲或缺拍。

每個實驗至少另存：獨立培養／樣本 ID、條件、取像日期、原始資料位置、實際或設定取像間隔、影像尺寸、像素尺度校正、曝光／對焦設定、異常與缺幀紀錄。原圖、原片及原始人工標註留在資料儲存位置，Git 只保存程式和必要範例。推送成功不等於實驗資料已備份。

## 2. 照片合併與分析影片

1. `flatten_buckets_gui.py` 可把分桶照片複製到單一資料夾。選「複製」，目的地放在來源樹以外；保留原始分桶結構。
2. 用 `images_to_video_gui_2.py` 自然排序照片並輸出 MP4/AVI。先確認順序、重名處理、影格總數及首末照片。自然檔名順序不是拍攝時間正確的證據。
3. 記錄選取影格範圍、FPS、輸出尺寸、留黑邊與裁切設定。為維持原始座標，分析輸入優先保留原尺寸並關閉形心裁切。
4. `video_timescaler_gui.py`、`mp4_compress.py`、`video_to_gif.py` 用來另做展示副本。變速會重複／捨棄影格；壓縮會改變像素；縮放與裁切改變尺度和座標。這些副本不能直接沿用原資料的時間與空間校正。

完整操作與原有覆蓋行為見 [影片工具說明](../video_tools/README.md)。

## 3. 啟動 V2，先檢查前中後片段

以下 PowerShell 命令假設從 **SpaceCellCam 儲存庫根目錄**開始，`python` 指向已備妥的分析環境：

```powershell
Set-Location .\plasmodium\plasmodium_analysis_v2
if (-not (Test-Path .\user_config.py)) {
    Copy-Item .\user_config.example.py .\user_config.py
}
python .\run_ui.py
```

本機既有環境可用 `C:\Users\chen8\.conda\envs\lbr\python.exe` 取代 `python`，在 PowerShell 引號路徑前加 `&`。不要覆蓋已存在的 `user_config.py` 或 `ui_settings.json`。兩個啟動程式都會匯入 `user_config.py`，新下載的專案必須先建立它。

在 UI 選擇實際輸入影片與輸出位置。UI 可能載入先前保存的 `ui_settings.json`，所以每次都要核對畫面上的值。若不使用 UI，先編輯 `user_config.py`，再執行 `python .\run_spyder.py`。

| 設定 | 現有預設／含義 | 每個實驗須確認 |
|---|---|---|
| `run_mode` | `validation_all` | 先看前／中／後抽樣輸出 |
| `frame_interval_sec` | `10.0` | 真實取像間隔；不是影片播放 FPS |
| `pixel_size` | `1.0` | 未做尺度校正時以 px、px² 報告，不把 1.0 當成 mm/px |
| `dish_center`、`dish_radius` | `(1280, 920)`、`680` px | 依輸入影像尺寸及培養皿位置重新確認 |
| `cpu_workers`、`parallel_batch_frames` | `0`、`0` | 0 代表自動；效能不足可手動降低 |
| `memory_fraction` | `0.55` | 可用記憶體的配置比例 |
| `validation_section_frames` | `300` | 每個抽樣區段影格數 |
| `validation_video_scale` | `0.5` | 疊圖輸出縮放；分割仍用原解析度 |

`validation_all` 會產生前中後片段、對應分析數值與圖。逐一檢查皿緣／反光誤抓、淡枝漏抓、影像晃動、失焦、缺幀、遮罩跳動及 ROI。它是開發期的視覺檢查模式，名稱中的 validation 不等於已通過準確度驗證。

## 4. 全片分析與結果追溯

輸入、校正和開發參數確認後，使用 `formal_only`；需要同時輸出三段檢查時使用 `formal_with_validation`。`formal` 表示全片分析的執行範圍，不代表科學驗證已完成。`contraction_roi` 是局部亮度／追蹤分析；不要僅以其週期結果宣稱管內流速或生理收縮。

每次執行預設建立 `runs/<時間>_<模式>/`，保存：

- `run_manifest.json`：設定、環境、效能配置、engine hash、產物與狀態。
- `run_mode_indicator.csv`：本次模式及遮罩方法。
- 該模式的 CSV、曲線、遮罩疊圖及檢查資料；模式不同，檔案集合也不同。

以 manifest 的 `completed`／`failed`／`cancelled` 及日誌判斷本次執行狀態，不能只看檔案是否存在。保存 Git commit、完整設定、輸入資料識別與校正紀錄；單一 engine hash 不足以代表所有模組版本。

## 5. 開發期人工比較

V2 的三個模式依序使用：

1. `manual_validation_export`：從影片抽取影像與目前自動 mask，建立新 run。
2. `manual_validation_annotate`：指定該 export run 的 `resume_run_directory`，進行人工編修。
3. `manual_validation_score`：使用相同 `resume_run_directory` 比較結果。

範例設定片段（路徑需換成實際 export run）：

```python
run_mode="manual_validation_annotate",
resume_run_directory=Path(r"D:\experiment-results\runs\actual_export_run"),
```

這套流程適合開發期找問題與比較。看到自動 mask 後再修正的標註，不應直接當成獨立盲測真值。前中後相鄰影格也不能當成獨立生物重複。

## 6. 獨立標註、凍結與最終驗證

使用 **V2 內的** [publication_validation_system](../plasmodium_analysis_v2/publication_validation_system/README.md)，不要與上層同名歷史資料夾混用。先閱讀其標註規範，建立本機 `videos.csv`，確認每列的影片、角色、獨立培養 ID、條件、培養皿幾何、時間及尺度。

在 `plasmodium/plasmodium_analysis_v2/publication_validation_system` 內執行：

```powershell
# 先建立並填寫 videos.csv；不覆蓋既有清單。
if (-not (Test-Path .\videos.csv)) {
    Copy-Item .\videos_template.csv .\videos.csv
}
python .\prepare_dataset.py --videos .\videos.csv --role validation
```

標註者從原始影像獨立繪製；保留各標註者與重複標註，完成爭議裁決後才產生共識 mask。開發資料只用於調參；測試資料應以獨立培養／影片分開，凍結前不查看測試結果。若查看 test 結果後調參，原 test 必須重新歸為開發資料。

**目前先不要把這條路徑宣稱為已驗證的正式流程。** `algorithm.status` 仍為 `development_not_frozen`；此外 `generate_predictions.py` 建立的 temporal 項目沒有帶入培養皿幾何，而 V2 的補償函式需要這些資訊估計平移，因此驗證輸出不能直接宣稱與主分析補償路徑等價。須先解決並測試此差異，再凍結來源與參數。

完成上述開發與資料前提後，既有工具的順序為：

```powershell
python .\freeze_algorithm.py --confirm-development-complete
python .\prepare_dataset.py --videos .\videos.csv --role test
python .\generate_predictions.py --role test
python .\evaluate_validation.py --role test
```

`--allow-development` 只供開發測試，不能繞過正式驗證資格。報告需分別檢查面積、邊界、clDice／骨架、動態變化及標註者一致性；面積吻合不等於細管拓撲或生物學解釋正確。

## 7. AI pilot 的使用邊界

AI pilot 與 V2 分開；先由 `pilot_config.example.json` 建立本機 `pilot_config.json`，設定影片資料夾。`prepare_dataset.py` 按整支影片分配 train/val/test；每條件至少三支影片，影片拆分本身仍不保證獨立培養，需另查實驗設計。資料清單一旦使用，勿因增減影片而任意重建分組。

弱標註與 SAM2 初稿只能輔助人工。尤其目前 `review_masks.py` 的 `n`／`p` 也會自動保存到 `masks_verified`，所以「存在該資料夾」不代表逐張確認完成。須另外記錄確認者、時間與狀態，正式使用前逐張審查。未找到本專案專用模型已完成訓練及獨立驗證的依據，不能把 AI pilot 當成替代主分析的方法。

## 尚待完成的研究與開發工作

1. 對齊主分析與獨立驗證的補償路徑與設定，建立相同輸入輸出的一致性測試。
2. 完善人工確認狀態；隔離草稿、自動保存與真值。
3. 驗證震動、旋轉、皿緣與曝光變化下的表現。目前 V2 暫時對齊 mask 後轉回各影格原座標，不是全部量測都在穩定座標系進行。
4. 完成時間／尺度校正、物種確認、獨立培養重複與最終盲測，再討論生物效應。

程式模組與開發檢查見 [DEVELOPMENT.md](DEVELOPMENT.md)。

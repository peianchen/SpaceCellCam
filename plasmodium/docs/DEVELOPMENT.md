# 開發與維護指南

更新：2026-11-14。主要維護線為 `plasmodium_analysis_v2`，獨立工具放在 `video_tools`，AI 研究支線放在 `ai_segmentation_pilot`。既有檔名與路徑保留，避免破壞歷史執行或結果追溯。

## 取得專案後

從 SpaceCellCam 儲存庫根目錄，在 PowerShell 或已啟用環境的終端執行：

```powershell
Set-Location .\plasmodium\plasmodium_analysis_v2
if (-not (Test-Path .\user_config.py)) {
    Copy-Item .\user_config.example.py .\user_config.py
}
python .\run_ui.py
```

`python` 必須指向已有依賴的環境。本機可在 Anaconda Prompt 使用 `conda activate lbr`；PowerShell 也可用 `& 'C:\Users\chen8\.conda\envs\lbr\python.exe'` 取代 `python`。新環境依賴列於 `requirements-lbr.txt`，其中包含 PyQt5、OpenCV、NumPy、pandas、SciPy、scikit-image、matplotlib、tqdm、psutil 和 pytest。需求檔是版本下限清單，並非完整鎖定環境；重現研究時另存實際版本。

`run_ui.py` 和 `run_spyder.py` 都匯入 `user_config.py`。UI 的保存設定在 `ui_settings.json`，可能覆蓋介面的初始值。兩者都是本機設定，勿用範例覆蓋正在使用的設定。不要直接執行 `plasmodium_v2/engine.py`。

## 模組責任與改動位置

| 位置 | 責任 | 修改後應檢查 |
|---|---|---|
| `config.py`、`user_config.py` | 型別化設定、有效值、日常參數 | 設定檢查、UI 轉換、resume 行為 |
| `ui_app.py`、`ui_worker.py` | 表單、背景程序、日誌、停止 | 設定保存、開始／停止、程序清理 |
| `runner.py`、`run_context.py` | 分析生命週期、獨立輸出、manifest | 成功、失敗、取消、續接紀錄 |
| `performance.py` | CPU 與可用記憶體的工作配置 | worker／batch 邊界與低記憶體模式 |
| `segmentation.py`、`mask_utils.py` | 單影格 HSV/chroma/vesselness 與 mask | 本體、淡枝、皿緣、硬上限、拓撲 |
| `temporal.py` | 前後影格投票、平移補償、halo | 批次邊界、位移閾值、原遮罩保留 |
| `manual_validation.py` | 開發期抽樣、人工比較套件 | 同一方法輸出、邊緣影格、續接 |
| `validation_api.py` | 獨立驗證使用的函式出口 | 與主分析的參數及行為一致性 |
| `engine.py` | 串流／批次、量測、ROI、圖表及模式整合 | 影格數、輸出模式、時間／尺度、長片記憶體 |
| `publication_validation_system/` | 盲測、凍結、預測、評分 | 來源鎖、資料角色、獨立單位與評分資格 |

主呼叫關係為 `run_ui.py → ui_app/ui_worker → runner.run_analysis → engine.run`；無 UI 路徑為 `run_spyder.py → runner.run_analysis`。UI 使用 multiprocessing 背景程序。新增設定應通過 `AnalysisSettings`、UI 讀寫、runner/runtime 與執行紀錄，而非只加到 engine 的區域變數。

## 測試入口

以下命令都在 `plasmodium/plasmodium_analysis_v2` 執行，輸入為程式建立的合成測試資料，並非真實實驗驗證：

```powershell
python -m pytest -q tests
python -m tests.run_end_to_end_smoke
python -m tests.run_contraction_stream_smoke
python .\publication_validation_system\synthetic_smoke_test.py
```

依改動選擇需要的測試；一般文件修改不必重跑所有影片流程。主要覆蓋點：

- `test_core.py`：分割、時間處理、配置與核心行為。
- `test_manual_validation.py`、`test_resume_workflow.py`：抽樣及 annotate/score 續接。
- `test_streaming_contraction.py`：分批與懶載入 contraction 路徑。
- `test_ui.py`：Qt 表單與效能預覽；使用 offscreen，不等於人工實際點擊驗收。
- `test_roi_ui.py`：ROI 命名及無終端輸入的行為。
- smoke 程式：合成資料端到端接線及輸出；不能證明實驗 mask 準確度。

AI pilot 的基礎測試可從 V2 目錄切換後執行：

```powershell
Set-Location ..\ai_segmentation_pilot
python -m pytest -q tests
```

這裡測試的是影片分組與合成影像的配準，不代表 SAM2、GPU、模型訓練或真實資料驗收。AI 推論使用另建的 `.venv_ai` 及 `requirements-ai.txt`；保留 `lbr`，不要為 AI 工作直接改動主分析環境。

## 獨立驗證與來源鎖

V2 的 `publication_validation_system/config.json` 指向 `../plasmodium_v2/validation_api.py`。凍結流程將 `validation_api.py`、`segmentation.py`、`temporal.py`、`mask_utils.py` 組合為來源 hash，並記錄預測器與參數 hash。相關來源或參數更動後需重新完成開發與凍結；不可沿用舊 lock 解釋新結果。

上層 `plasmodium/publication_validation_system` 指向舊 optimized 單檔，是另一套歷史路徑。兩份 config、輸出根目錄與來源鎖不可互換。一般執行 manifest 的 engine hash 和 publication 的組合來源鎖也不是同一件事。

目前 `generate_predictions.py` 的 temporal items 沒有帶入 `dish_center`／`dish_radius`。補償雖預設啟用，但缺少幾何會使平移估計退回未補償路徑；正式主流程與 `manual_validation.py` 會帶入幾何。修正時應加「同一影格、同一參數、同一 halo、同一補償」的跨入口一致性測試，再進行新的來源凍結。

## 開發優先順序

1. **驗證路徑一致性**：先對齊上述補償資訊，再比較分割參數與 halo，避免用不同演算法的驗證數字替主流程背書。
2. **人工標註狀態**：AI `review_masks.py` 在前後翻頁時會自動保存到 `masks_verified`；增加明確確認狀態及追溯前，不能只依資料夾判斷真值。
3. **量測座標系**：V2 補償是投票時暫時平移對齊，最終 mask 轉回每格原座標；相鄰面積變化、前緣速度和 ROI 仍可能受晃動影響。旋轉與統一量測座標需另立範圍驗證。
4. **可重現設定**：保留完整設定、commit、依賴版本、輸入識別、尺度及時間校正。不要讓新增預設值默默改變舊實驗的重跑條件。
5. **小步拆分 engine**：模組化時先固定接口與測試；不要順便重命名 CSV 欄位、改 ROI 座標或移走歷史程式。

## 版本控制與資料保存

- 提交程式、測試、依賴清單、設定範例、操作文件與必要的小型合成 fixture。
- 本機設定、原始照片／影片、實驗 mask、衍生量測、runs、模型 checkpoint、虛擬環境、建置及暫存保持在本機資料流程，不混入一般程式提交。
- 忽略規則不是刪除指令；檔案已被 Git 追蹤時，新增忽略規則也不會自動停止追蹤。
- 推送前先看遠端差異與待提交清單；先整合遠端，禁止用強制推送覆蓋另一邊工作。
- 需要搬移資料或改輸出格式時，先提供相容方式與核對清單；保持既有分析 run 可追溯。

使用者操作順序、實驗證據限制與輸出解讀見 [CURRENT_WORKFLOW.md](CURRENT_WORKFLOW.md)。

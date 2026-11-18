# 黏菌影像處理與分析

更新：2026-11-18。這裡是 SpaceCellCam 的影像整理、V2 分析、人工驗證與 AI 標註試驗區。
目前建議的分析／開發入口是 **`plasmodium_analysis_v2/run_ui.py`**。

## 從哪裡開始

1. 先讀 [完整處理流程](docs/CURRENT_WORKFLOW.md)：取像紀錄、SD 備份、照片排序、影片製作、抽樣檢查、全片分析與人工驗證。
2. 依 [V2 操作說明](plasmodium_analysis_v2/README.md) 啟動介面。全新下載專案後，先把 `user_config.example.py` 複製為 `user_config.py`，再設定自己的影片路徑及校正值。
3. 改程式前讀 [開發指南](docs/DEVELOPMENT.md)；資料保存與 Git 範圍見 [版本管理規則](docs/REPOSITORY_POLICY.md)。
4. 暗光細網與 RPM 震動的新候選方法見 [segmentation_lab](segmentation_lab/README.md) 與 [第二輪版本摘要](segmentation_lab/milestones/20261118-round2.md)。該工具可獨立測試；目前尚未接管 V2 的正式量測。

## 目前程式分類

| 類別 | 位置／入口 | 定位 |
|---|---|---|
| 影像前後處理 | [video_tools](video_tools/README.md) | 合併照片、轉影片、變速、壓縮、GIF；獨立執行 |
| 現行主線 | [plasmodium_analysis_v2](plasmodium_analysis_v2/README.md) | PyQt 操作、分批分析、結果紀錄、人工工作包 |
| V2 發表驗證 | [V2 publication_validation_system](plasmodium_analysis_v2/publication_validation_system/README.md) | 獨立抽樣／凍結／預測／評估；目前仍為 development_not_frozen |
| AI 標註試驗 | [ai_segmentation_pilot](ai_segmentation_pilot/README.md) | 配準、按影片分組、SAM2 草稿／V2 弱標註；未接入主線，未建立專用模型訓練完成證據 |
| 分割與震動改善候選 | [segmentation_lab](segmentation_lab/README.md) | 局部細網、食物排除、獨立分割測試與保守配準；尚無人工真值準確率 |
| 舊版發表驗證 | [publication_validation_system](publication_validation_system/README.md) | 對應根目錄 optimized 單檔；不要與 V2 的輸出包混用 |
| 舊整合單檔 | `plasmodium_brightfield_analysis_optimized.py` | 保留以重現歷史工作；操作紀錄見 [歷史說明](docs/LEGACY_OPTIMIZED_WORKFLOW.md) |
| 其他歷史版本 | `plasmodium_brightfield_analysis*.py`、`plasmodium_sample*.py` | 不自動取代 V2；名稱含 formal／complete／optimized 不等於已通過科學驗證 |
| 參數與診斷工具 | `mask_parameter_sweep_new.py`、`vesselness_recovery_sweep.py`、`tune_*.py`、V2 `tools/` | 開發用參數搜尋與跨影片檢查；輸出留在本機 |
| 舊人工 mask 工具 | `prepare_draft_masks.py`、`redraw_rear_masks_from_manual.py`、`run_manual_mask_validation.py`、`validate_temporal_candidate_against_manual.py` | 保留依賴和原路徑；草稿不能直接當最終真值 |

## 輸出與證據界線

- `validation_all` 是前／中／後段視覺與流程檢查；`formal_only` 是全片執行範圍。兩者名稱都不表示已證明準確率。
- 真實取像間隔與影片播放 FPS 分開；像素尺度未校正時用 px、px²。變速或裁切展示副本不能直接沿用原始時間／座標設定量化。
- V2 目前以傳統影像處理為主。震動補償、正式／發表驗證一致性及人工標註狀態仍有已知缺口，見開發指南。
- 菌種應依實驗紀錄與鑑定確認。歷史文字中的 *Physarum polycephalum* 不能證明目前培養物就是該物種。
- Git 保存程式與流程文件；原始影像、標註、結果、模型及本機設定另行保存，詳見版本管理規則。

## 歷史文件與查核

- [2026-11-13 程式能力查核](PROGRAM_ANALYSIS_20261113.md)：當日來源與測試快照；後續改版須重新驗證相關結論。
- [舊方法與文獻解讀](docs/LEGACY_ANALYSIS_NOTES.md)：原首頁內容保留，供方法背景與歷史比較。
- [舊圖文 HTML](PLASMODIUM_ANALYSIS_REPORT.html)：歷史解讀，不作目前版本的驗證報告。
- [舊 temporal 候選說明](TEMPORAL_FORMAL_CANDIDATE_README.md)：歷史候選支線。
- [本次整理與測試紀錄](docs/ORGANIZATION_20261114.md)。

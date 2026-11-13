# 專案整理與查核紀錄

日期：2026-11-14。範圍：SpaceCellCam 儲存庫、影片工具、V2、AI pilot、歷史分析與驗證程式。

## 遠端與本機的處理

整理前本機 main 為 `198a8e3`；取得遠端後，origin/main 為 `f78a64a`，本機沒有額外已提交版本、遠端多一筆。
該筆只更新根 README 並新增中英文 ENGINEERING 文件，已以 fast-forward 整合；沒有覆寫遠端歷史。
本機未提交修改先保留差異備份，並建立本機備份分支 `codex/backup-before-organize-20261114`。

## 整理內容

- 將原先未追蹤的 V2、AI pilot、影片工具、測試、歷史單檔及開發工具納入版本控制。
- 保留所有程式原路徑，未重構演算法；以 README 和流程文件建立現在／歷史／試驗支線的分工。
- 現行主入口為 V2 `run_ui.py`；`OFFICIAL_ANALYSIS_WORKFLOW.md` 改為目前流程索引。
- 舊首頁方法解讀及舊 optimized 操作文件保存至 `docs/LEGACY_ANALYSIS_NOTES.md`、`docs/LEGACY_OPTIMIZED_WORKFLOW.md`。
- 本機 `user_config.py`、`ui_settings.json`、`pilot_config.json` 保留，Git 提供可攜式 example。
- 兩份影片清單範例改用佔位路徑；原本內容另存本機備份，並在沒有既有 `videos.csv` 時保留為本機清單。
- 排除快取、計畫、簡報生成物、runs、參數搜尋結果、診斷輸出、AI dataset、模型、虛擬環境與大型媒體。忽略不等於刪除，也不等於資料備份。
- 保留本機原先修改的 legacy publication `config.json`，並連同所需來源納入提交；其狀態仍為 `development_not_frozen`。

## 本次實際檢查

| 檢查 | 結果與範圍 |
|---|---|
| V2 既有 pytest | 11 passed；包含核心、人工工作包、續接、串流與離屏 UI 測試 |
| AI pilot 基礎 pytest | 2 passed；影片分組及合成配準，未跑 SAM2/GPU 推論 |
| 只含待提交內容的獨立副本 | 由 example 建立設定後，run_ui/run_spyder 皆可載入 |
| 獨立副本 V2 合成影片 smoke | 通過，manifest 為 completed；64×64 疊圖影片可解碼 |
| 獨立副本 V2 publication smoke | 30 張合成樣本完成流程；狀態保持 NOT_ESTABLISHED |
| 待提交來源與設定 | Python AST／JSON 可解析，新增檔案均小於 1 MB，未混入媒體、模型或資料目錄 |
| 來源與本機設定保留 | 主模組、影片工具及本機設定的保留 hash 核對通過 |
| 文件與排除範圍 | 入口連結及副本啟動步驟檢查，另做一次獨立唯讀審查，未發現阻擋問題 |

V2 第一次 pytest 有 6 個 setup errors，原因是指定暫存父目錄不存在；建立測試目錄後 11 項全部通過，沒有為此修改程式。
`git diff --check` 提示部分匯入的既有程式末尾空行與歷史 Markdown 換行空白；為保留原始程式內容，本次不做全面格式改寫。

本次沒有安裝／升級依賴、燒錄或測試實體相機、重新建置韌體、執行全長實驗影片、修改標註或訓練模型。
合成 publication 測試使用程式自己的 temporal 預測建立參考遮罩，驗證的是接線與自我一致性，不是人工真值準確率。

## 接續工作

已知補償路徑、人工確認狀態與量測座標限制列於 [開發指南](DEVELOPMENT.md)。先完成這些一致性與證據問題，再進行獨立生物驗證；本次整理及推送不表示缺口已修復。
操作從 [目前流程](CURRENT_WORKFLOW.md) 開始，資料保存依 [版本管理規則](REPOSITORY_POLICY.md)。

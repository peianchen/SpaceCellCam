# 影像整理與影片工具

於 2026-11-13 從 `C:\Users\chen8\OneDrive\博士\太空生醫\video` 納入本專案。
7 支 Python 與 3 份 PyInstaller 設定均原樣複製，來源及 SHA-256 記錄於
[`IMPORT_MANIFEST.json`](IMPORT_MANIFEST.json)。原資料夾、4 部 MP4 以及 `build/`、`dist/` 保持原狀；打包成品和第三方程式庫未重複納入。

## 工具選擇

| 工作 | 程式 | 功能／版本定位 |
|---|---|---|
| 合併分桶照片 | `flatten_buckets_gui.py` | 遞迴收集影像到單一資料夾；可複製或搬移，重名時加流水號 |
| 照片轉影片 | `images_to_video_gui_2.py` | 功能較完整的入口；自然檔名排序、影格區間、尺寸、留黑邊、MP4/AVI、可選平滑形心裁切 |
| 分段變速 | `video_timescaler_gui.py` | 按來源影格區間加速／減速，輸出不含音訊 |
| 影片壓縮 | `mp4_compress.py` | FFmpeg 壓縮；可調 CRF、編碼器、音訊和縮放 |
| 影片轉 GIF | `video_to_gif.py` | FFmpeg 調色盤轉換；可調 FPS、寬度及截取時段 |
| 照片轉影片歷史版 | `images_to_video_gui.py` | 原始版本；現有同名 `.spec` 仍指向此版本 |
| 照片轉影片中間版 | `images_to_video_gui_1.py` | 加入輸出前重掃照片、MP4 失敗備援與執行緒清理 |

原始名稱保留，避免打包設定失去對應。`_2` 的額外裁切功能預設關閉；此分類依程式內容，不表示已完成各版本的實驗資料驗證。

## 啟動

可在 Spyder 選用既有 `lbr` 環境，開啟表中的程式按 Run。
也可在 PowerShell 執行以下範例，最後的檔名換成所需工具即可：

```powershell
Set-Location 'C:\Users\chen8\Documents\PlatformIO\Projects\SpaceCellCam\plasmodium\video_tools'
& 'C:\Users\chen8\.conda\envs\lbr\python.exe' '.\images_to_video_gui_2.py'
```

執行需求為 Python、PyQt5、OpenCV、NumPy，列於 [`requirements.txt`](requirements.txt)。
匯入時確認既有 `lbr` 可載入 PyQt5 5.15.11、OpenCV 4.12.0、NumPy 2.3.5；本次未安裝或升級套件。
壓縮另需 PATH 中的 `ffmpeg`、`ffprobe`，GIF 需要 `ffmpeg`。壓縮／GIF 程式可透過 Qt 選檔，找不到適用 Qt 時改用終端輸入。

## 與分析專案的銜接

1. 若照片分散在 `b000`、`b001` 等資料夾，可先用合併器選擇「複製」，輸出到來源樹以外的新資料夾，避免再次收集自己的輸出。同名照片會重新命名，合併後須確認檔名順序與實際取像順序一致。
2. 使用照片轉影片工具產生 MP4/AVI，記錄原始取像間隔、取用影格範圍及任何尺寸／裁切設定。
3. 到 [`plasmodium_analysis_v2`](../plasmodium_analysis_v2/README.md) 的分析介面選取影片，另行設定真實取像間隔與像素尺度。播放 FPS 不等於實驗取像頻率。
4. 變速、壓縮與 GIF 適合另做展示副本。變速會丟棄或重複影格，裁切／縮放會改變座標與尺度，壓縮也會改變像素；不能直接沿用原始資料的時間／空間校正進行量化。

這些工具是獨立的前後處理程式，目前沒有接入 V2 分析介面按鈕或自動流程。

## 原有操作行為

- 合併器的「搬移」會移走來源照片；需要保留原始資料時使用「複製」。
- 壓縮及 GIF 的 `OVERWRITE = True` 會覆蓋同名輸出；處理前可改為 `False`，或先使用獨立副本。兩者輸出在輸入檔案旁，分別加 `_compressed.mp4`、`_gif.gif`。
- GIF 使用輸入檔旁的 `<檔名>_palette.png` 作暫存，成功後刪除該檔；避免在該名稱已有重要檔案的資料夾執行。
- 變速表使用 1 起算且含終點的來源影格，例如 `1-300@0.5, 301-900@2`；未指定區間維持 1 倍速。

## 打包設定

保留 `flatten_buckets_gui.spec`、`images_to_video_gui.spec`、`video_timescaler_gui.spec`。
若日後需要重新打包，在本資料夾使用已具備 PyInstaller 的環境執行，例如：

```powershell
python -m PyInstaller images_to_video_gui.spec
```

此命令打包的是原始 `images_to_video_gui.py`，不是 `_2`；本次未重新建置 EXE。
`build/`、`dist/` 與 Python 快取已由本資料夾的 `.gitignore` 排除。

## 本次查核範圍

檔案完整性、Python／spec 語法及現有環境載入結果見 [`IMPORT_CHECKS.md`](IMPORT_CHECKS.md)。
未執行真實實驗資料合併、搬移、轉檔或重新分析。

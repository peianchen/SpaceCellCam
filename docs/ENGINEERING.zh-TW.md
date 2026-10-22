# SpaceCellCam: 工程說明

[README](../README.md) · [中文](ENGINEERING.zh-TW.md) · [English](ENGINEERING.md)

以 ESP32-S3 與 OV5640 AF 控制定時照明、對焦重用、JPEG 擷取與 microSD 寫入，讓實驗取像流程可觀察、可追查。

## 目的與用途

長時間生物觀察需要穩定的取像順序與可保存的影像。這個韌體將相機、LED、記憶體和 SD 儲存排成明確週期，並透過日誌區分對焦、取像、待機與寫檔問題。

## 架構與流程

```text
WAKE + LIGHT -> FOCUS + JPEG -> PSRAM + STANDBY -> SD + VERIFY
```

### 喚醒與照明

目前設定目標週期為 10 秒、XCLK 為 20 MHz，先喚醒感測器並開啟 LED，等待 300 ms 讓照明與曝光穩定。這些是設定值。

### 對焦與擷取

第一次使用中央區域自動對焦並鎖定；後續擷取重用焦點，丟棄一張過渡影格，再取得 QSXGA JPEG。

### 歸還緩衝區

先將 JPEG 複製到 PSRAM，再歸還 camera frame buffer，讓相機能進入 standby，之後才處理 SD 寫入。

### 寫入與檢查

影像先寫暫存檔，再重新命名並確認檔案大小。每個 bucket 保存 500 張；開機時回復編號以避免覆寫既有影像。


## 設計取捨

### 重用焦點的取捨

固定場景可省去每次重新搜尋焦點，但目標距離變動時可能失焦，需以照片與日誌一起判讀。

### 記憶體換取流程解耦

PSRAM 的 JPEG 副本增加暫時記憶體需求，讓 SD 寫入不必延長 LED 點亮及相機活躍時間。

### 保存失敗的線索

對焦失敗、frame timeout、短寫入與大小不符各有日誌，方便定位問題發生在哪一層。


## 結果與證據

### 可確認的設定

公開 README 與韌體描述 10 秒週期、focus-lock reuse、sensor standby 及暫存檔驗證流程。

### 結果判讀方式

使用 reuse locked focus、standby enter/wake 與 SD 驗證紀錄，對照實際照片判斷各階段是否完成。

### 尚未建立的結論

本頁沒有新增取像實驗，不以程式設定推論實際準時率、功耗、光學品質或長時間資料完整率。

## 重現方式

1. 準備 XIAO ESP32S3、OV5640 AF、microSD 與穩定供電，依 repository 的 platformio.ini 建置。
2. 記錄首次及後續擷取日誌，確認 AF 搜尋與重用路徑不同。
3. 固定場景後改變目標距離，比對照片清晰度與日誌；另檢查 SD 檔案大小、編號及重新開機後是否覆寫。

## 限制

standby／wake 時的 VCM 聲音不等於重新對焦；ESP32-S3 內部溫度也不是 OV5640 或馬達溫度。專案名稱不表示已完成太空環境或在軌驗證。

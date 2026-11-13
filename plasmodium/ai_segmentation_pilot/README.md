# Plasmodium AI Segmentation Pilot

這個資料夾是獨立的 AI mask 可行性實驗，不修改
`plasmodium_analysis_v2` 或舊版單檔分析程式。

更新提示（2026-11-14）：新下載專案後，先將 `pilot_config.example.json` 複製為
`pilot_config.json`，填入自己的影片目錄與幾何設定；已存在的本機設定不覆蓋。
整個 `dataset/`、模型與虛擬環境不隨 Git 推送，需另行保存。
完整處理順序及限制見 [目前流程](../docs/CURRENT_WORKFLOW.md)。

**現存限制：** `review_masks.py` 的 `n/p` 翻頁也會將當前草稿保存到
`masks_verified`；資料夾名稱不是人工確認證據。使用前必須逐張審查並另存確認紀錄，
在確認狀態改造完成前，不得僅憑該目錄存在檔案就投入訓練或最終評估。
下文的 GPU／SAM2 環境資訊與 120 張規模屬既有本機工作紀錄，不是新下載專案後的保證或本次新驗收結果。

## 設計原則

1. 先以培養皿邊緣校正相機平移與小角度旋轉。
2. 所有原始影片都納入資料分布，但只標註代表幀。
3. train / validation / test 以「整支影片」分開，避免相鄰幀洩漏。
4. AI 初稿與人工確認 mask 分開保存；未確認的 mask 不當成 ground truth。
5. 最終模型除了 Dice/IoU，也檢查細管骨架、連通性、皿緣假陽性與逐幀抖動。

## 第一階段資料集

預設從每支影片的 early / middle / late 各抽 5 張，共 120 張。輸出：

```text
dataset/
  images/                 配準後的原始解析度 PNG
  masks_draft/            SAM2 未確認初稿
  masks_weak_v2/          現有 V2 方法產生的弱標註
  masks_verified/         人工確認後才能放入訓練的二值 mask
  previews/               每支影片的抽樣接觸表
  dataset_manifest.csv    來源、frame、split、配準矩陣與品質資訊
```

執行：

```powershell
.\.venv_ai\Scripts\python.exe prepare_dataset.py
.\.venv_ai\Scripts\python.exe validate_dataset.py
```

## AI 環境

`lbr` 保留給正式分析；AI 套件安裝在 `.venv_ai`。目前已確認：

- RTX 4060 Laptop GPU、CUDA 可用。
- `torch 2.13.0+cu126` 與 SAM2 可載入。
- 通用 SAM2 對早期團塊偶爾可用，但對後期細網容易漏抓或填入框內背景，
  因此不得直接當 ground truth 或批次正式結果。

## 建議標註流程

先由正式 V2 spatial mask 產生弱標註：

```powershell
.\.venv_ai\Scripts\python.exe generate_v2_weak_labels.py --dataset dataset
```

再人工校正。左鍵補、右鍵擦、`s` 儲存、`n/p` 切換、`+/-` 改筆刷、`q` 離開：

```powershell
.\.venv_ai\Scripts\python.exe review_masks.py --dataset dataset --split train
```

校正完成的檔案存入 `dataset/masks_verified`。只有這個資料夾的 mask 才能用於
專用 semantic segmentation 模型訓練。建議先完成 train 代表幀，再完成 val/test；
不得從同一支影片拆幀到不同 split。

## SAM2 對照（僅診斷）

```powershell
.\.venv_ai\Scripts\python.exe compare_sam2_prompts.py `
  --checkpoint checkpoints\sam2.1_hiera_small.pt --dataset dataset --row 0
```

比較圖在 `dataset/previews/sam2_prompt_comparison_row0.jpg`。本 pilot 的 go/no-go
結論是：SAM2 可當互動式輔助，但目前不能取代人工確認或專用模型。

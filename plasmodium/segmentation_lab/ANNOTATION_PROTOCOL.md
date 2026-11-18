# 人工遮罩確認與評估

本工具把「演算法草稿」、「人工已確認」與「獨立盲測」分開記錄。現有研究資料不會被覆寫；請為每次標註建立新資料夾。沒有完整人工確認時，評估結果固定為 `status=awaiting_human_confirmation`、`metrics=null`。

## 遮罩代表什麼

前景是影像中可看見的黏菌組織，包括團塊、薄片及能辨認邊界的淡色細管。保留網路中可見的背景孔隙。排除培養皿、反光、背景、灰塵，以及可辨識的非黏菌中央食物／接種物；覆在食物上且能辨認的黏菌仍包含。此工具要求明確採用 `exclude_food` 政策。邊界不確定時需記錄並由研究者裁決，不用演算法填滿整片網路來代替判斷。

## 建立工作包

在專案根目錄執行（範例路徑請換成實際本次輸出）：

```powershell
python -m segmentation_lab.annotations export benchmark_run/samples.json annotation_round1 --mode assisted
```

也支援 `samples.csv`。影像／遮罩路徑以 samples 檔案所在目錄為基準。工作包包含：

- `images/`：複製的原解析度、原座標影像。
- `draft_masks/`：可供修改的草稿；無草稿或 `--mode independent` 時為全黑空白模板。
- `confirmed_masks/`：明確確認後保存的版本，以內容雜湊命名，保留歷次確認版本。
- `evaluation_predictions/`：僅供評分的演算法預測。獨立標註者不可查看此目錄。
- `annotations.json`：原圖／預測雜湊、草稿來源、狀態及每次確認的追溯紀錄。

`assisted` 指人看過演算法草稿再修正，只能作開發期參考。`independent` 指從原圖獨立繪製，不看任何演算法結果。宣告來源並不能證明盲測確實執行；研究者須另外保管標註者、分工、原始實驗與資料拆分紀錄。不得把 assisted 工作包改標 independent。

## 修改與明確確認

可用本機編輯器在原圖上補畫／擦除，保存仍維持原解析度：

```powershell
python -B -m segmentation_lab.mask_editor --package annotation_round1 --sample SAMPLE_ID
```

左鍵補、右鍵擦；`[`／`]` 調整筆刷；`+`／`-`／滾輪縮放，方向鍵或 I/J/K/L 平移，`0` 全圖；`V` 疊圖、`O` 原圖唯讀、`B` 二值圖，`Z` 撤銷。`S` 只存草稿，`Esc` 離開且不自動保存。工具檢查外部修改衝突。已做無視窗座標／保存測試，實際桌面 GUI 顯示與手動操作尚未驗收。

也可使用外部影像編輯工具，另存原解析度單通道 PNG：前景 255、背景 0，不旋轉、不用彩色或反鋸齒灰階。兩條路都不會自動確認真值。

僅存草稿，不代表完成：

```powershell
python -m segmentation_lab.annotations draft annotation_round1 SAMPLE_ID edited_mask.png
```

只有實際檢查過的標註者可執行以下命令。`--reviewer` 填自己的識別，`--source` 必須和建立工作包時相同；不得由代理代填真人確認。

```powershell
python -m segmentation_lab.annotations confirm annotation_round1 SAMPLE_ID edited_mask.png --reviewer YOUR_REVIEWER_ID --source assisted --exclude-food
```

確認記錄含確認者、UTC 時間、原圖 hash、mask hash、來源及排除食物政策。再次存草稿會撤銷當前確認資格，保留舊確認版本；須重新確認才能評分。原圖變動必須另建工作包。確認後的遮罩或預測如被改動，評分會拒絕。

## 評估與解讀

```powershell
python -m segmentation_lab.annotations evaluate annotation_round1 --out annotation_round1/evaluation.json
```

所有樣本都須有完整確認紀錄且雜湊一致；部分完成不會輸出部分準確率。輸出各樣本及各方法的 Dice、IoU、Precision、Recall、TP/FP/FN/TN 與像素面積誤差。各方法的樣本數會列出；方法失敗或缺輸出時不可忽略樣本數差異來比較平均分數。

空遮罩定義：兩者皆空，Dice/IoU 為 1；只有預測非空，Dice/IoU 為 0。預測為空時 Precision 定義為 1，真值為空時 Recall 定義為 1；仍須同時看 Dice、FP/FN。真值面積為 0 時相對面積誤差為 `null`，保留絕對像素差，絕不把除以零變成誤差 0。

`source_split=test` 不是獨立驗證證據。只在獨立人工來源、明確 `evaluation_role=independent_test`、不同於已記錄開發資料的 dataset ID 全部具備時，工具才標示 `requires_study_design_review`；`independent_test_eligible` 仍保持 false，由研究設計另行審查。既有 `development_seen` 資料不會因換檔名而升格。這裡的評分描述與記錄參考遮罩的一致程度，不自動證明生物效應或泛化能力。

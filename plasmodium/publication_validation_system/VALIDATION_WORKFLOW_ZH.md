# 論文級驗證執行流程

## 現在的狀態

- `output2.mp4` 已被看過並用於調參，只能列為 `development`。
- `config.json` 目前是 `development_not_frozen`，這是刻意的；開發完成前不要執行 freeze。
- 現在 temporal 參數是本輪選定的 `connect_radius=10`、`block_hole_max_area=1500`，仍屬開發參數。

## 開發完成後

1. 準備完整的 `videos.csv` 登錄表，不能省略曾用於開發的影片。
2. 至少準備 3 個獨立 test 生物重複；實際數量應依研究設計或 power analysis 增加。
3. 每支影片只能屬於 development、validation、test 其中一類，同一培養物不能跨類別。
4. 若還需要最後一次調整，只能在 validation 影片做；看過 test 指標後不能再調參。
5. 開發結束後執行：

```powershell
python freeze_algorithm.py --confirm-development-complete
```

6. 鎖定後才能建立 test 抽樣包：

```powershell
python prepare_dataset.py --videos videos.csv --role test
```

7. 兩位標註者只取得 `annotation_blinded` 內的全解析度原圖，依 `ANNOTATION_PROTOCOL.md` 獨立繪製。
8. 完成 primary 與 repeat 標註，再產生 adjudicated consensus。期間不要讓標註者看到預測。
9. 檢查完整性：

```powershell
python check_annotation_status.py --role test
```

10. 產生鎖定預測並評估：

```powershell
python generate_predictions.py --role test
python evaluate_validation.py --role test
```

## 判讀順序

1. 先看 `evaluation_status.json` 是否為 `ELIGIBLE_AND_PASSED`。
2. 看 `inter_rater_metrics.csv` 與 `intra_rater_metrics.csv`，確認人工 reference 本身可重現。
3. 看 `summary_by_method.csv` 的 video-clustered 95% CI，不用單張 frame 當獨立 n。
4. 分別看 front/middle/rear 的 `summary_by_phase.csv`，不能只報全部平均。
5. 看 `failure_gallery` 的所有最差案例，確認錯誤不是 rim、反光、內部空隙或淡支線的系統性偏差。
6. 面積 mask 過關不代表 skeleton、endpoints、branchpoints、vein 或 front speed 自動過關；各欄位要看自己的誤差表。

## 失敗後的規則

若 test 未達標但又要修改參數，這批 test 立即改列 development。重新 freeze 後，必須用新的獨立培養物建立新 test set；不能在同一批 test 反覆調到過關。

## Paper 中至少要報告

- development/validation/test 的影片數與獨立培養物數；
- 抽樣規則、每段幀數、是否連續視窗、排除首尾幀規則；
- 兩位標註者、repeat 比例、adjudication 方法；
- Dice、IoU、precision、recall、area error、Boundary F1、HD95；
- clDice、skeleton length、endpoint、branchpoint（若論文使用拓撲結論）；
- expansion/retraction 與 front-motion 誤差（若論文使用動態結論）；
- 以獨立影片/培養物做 cluster bootstrap 的 95% CI；
- Bland-Altman 圖、分期結果及 worst-case failure gallery；
- 鎖定程式與參數的 SHA-256、Python/OpenCV/scikit-image/scipy 版本。

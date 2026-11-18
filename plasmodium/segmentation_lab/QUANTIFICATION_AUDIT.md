# 原版與 V2 定量、時間處理查核

查核日期：2026-11-18。依目前工作區來源逐段確認實際入口與函式呼叫，沒有以舊報告代替現況；沒有修改原版、optimized 或 V2，也沒有在本次查核重跑研究影片。本文是內部方法與風險說明，不能代替人工真值或生物實驗驗證。

**核心結論：目前 V2 會把時間補回後的 mask 用於正式形態與相鄰變化量測；其運動補償只協助時間投票，並沒有把正式新增／退縮／速度計算改成共同座標。時間補回也可能把真實的短暫退縮補回。** 新 `segmentation_lab` 的單張分割與時間診斷是獨立比較工具，尚未接入舊正式定量入口。

## 1. 來源與目前有效入口

以下以 O、E、T 縮寫標示檔案與實際行號：

- O：[原版 plasmodium_brightfield_analysis.py](C:/Users/chen8/Documents/PlatformIO/Projects/SpaceCellCam/plasmodium/plasmodium_brightfield_analysis.py)。
- E：[V2 engine.py](C:/Users/chen8/Documents/PlatformIO/Projects/SpaceCellCam/plasmodium/plasmodium_analysis_v2/plasmodium_v2/engine.py)。
- T：[V2 temporal.py](C:/Users/chen8/Documents/PlatformIO/Projects/SpaceCellCam/plasmodium/plasmodium_analysis_v2/plasmodium_v2/temporal.py)。
- S：[V2 segmentation.py](C:/Users/chen8/Documents/PlatformIO/Projects/SpaceCellCam/plasmodium/plasmodium_analysis_v2/plasmodium_v2/segmentation.py)。

| 核對項 | 原版目前檔案設定 | V2 目前檔案設定 |
|---|---|---|
| 入口與預設模式 | `__main__`，`test_front_middle_rear`；只輸出前中後驗證片段，不執行全片正式量測。O:5832、5858–5884 | `run_spyder.py:5–11` → `runner.run_analysis` → `engine.run`（runner.py:21–48）；user_config.py:11 為 `validation_all` |
| 預設 validation 是否有 CSV | 不能把它當成全片正式分析；正式開關由 O:5871–5874 決定 | **有前中後片段的正式式 CSV 與圖**：E:8245–8248、8695–8749；E:8797–8798 結束本輪，不再執行全片。片段實際呼叫完整 temporal analyzer：E:3929–3965 |
| 分割來源 | 原版檔案內函式 | E:8007–8020 將 segmentation 與 temporal 函式重新綁定到模組；審查時間流程必須看 T，不能只看 engine 內舊重複定義 |
| 時間／尺度 | `FRAME_INTERVAL=10.0`、`PIXEL_SIZE=1.0`。O:5895–5899 | settings 取得 10.0、1.0（user_config.py:14–15；E:8169–8173）。UI 可覆寫設定，個別執行仍應讀其 manifest |
| 正式運算解析度 | 寬高各 0.5。O:5957–5960；validation 也會先縮圖再分割，O:5093–5109 | 運算 1.0；僅輸出影片縮為 0.5。E:8285–8289 |
| 時間處理 | 5 張／3 票補回，再 5 張／3 票平滑。O:5936–5950 | 5 張／4 票補回、最後平滑關閉（window=1）；開啟培養皿平移對齊。E:8254–8278、T:154–166 |
| 骨架／獨立管網 | body skeleton 開；獨立 vein/ridge 分析關閉。O:5961–5963 | 同樣 body skeleton 開、`VEIN_EVERY_N_FRAMES=0`。E:8290–8292 |
| 收縮 | `ROI=None`，不產生 ROI 灰階週期分析。O:5899、2798–2807 | `COMPUTE_CONTRACTION=True`，但 `CONTRACTION_ROIS=[]`；預設 validation 模式實際略過。E:8181–8186、8296–8309。`contraction_roi`／`complete` 才進入 ROI 選取流程（E:8133、8176–8179） |

原版與 V2 現有圖像結果不能只以「版本名稱」比較：正式／單張／validation、解析度、ROI、時間窗與 batch 都必須列明。`BASELINE_AUDIT.md` 的共同原解析度單張 baseline 不等於舊預設完整流程。

## 2. 真正送進量測的 mask

| 名稱 | 程式實際定義 | 可以解讀成甚麼 |
|---|---|---|
| `occupied_mask` | 單張分割選到的前景；現有來源沒有獨立食物排除。S:222–288，另見 BASELINE_AUDIT.md | 演算法選到的可見前景像素，仍可能含誤判；不是已證實的黏菌真值 |
| `clean_mask` | occupied 加上可選的小孔填補；目前主入口關閉填小孔，所以單張 clean=occupied | 為形態量測準備的前景；若日後開孔洞填補，面積與骨架語義會改變 |
| `envelope_mask` | 填滿 enclosed holes 的參考包圍區域；不是 convex hull | 占據外圍內的覆蓋範圍；不能當作可見組織面積 |
| `strict_mask` | temporal 前的單張結果；V2 已可能含黃色延伸與 vesselness 補回 | 名稱 strict 不表示純 HSV，也不表示人工確認 |
| `temporal_mask` | 單張結果與跨張候選投票補回的聯集，再保留最大連通區 | 使用前後時間資訊推定的前景；不是當張直接可見證據 |

非 temporal 路徑用 `_body_mask_layers` 回傳的 **clean** 做 region、body skeleton 與新增／退縮；獨立 vein 分析的限縮範圍用 occupied。O:1863–1892；V2 E:2183 起的相同處理段。

目前有效 temporal 路徑則直接把 `occupied=clean=temporal_mask`，另由它填孔得到 envelope：O:2674–2676；E:3117–3119。**因此 temporal CSV 的 `occupied_area` 已經包含時間推定補回，不能說成完全未修飾的當張觀察。** 原單張面積另存 `strict_hsv_area`、補回量另存 `temporal_added_area`（O:2683–2697；E:3126 起）。

## 3. 各指標到底在算甚麼

表中 M 指實際量測 mask：一般路徑為 clean；目前 temporal 正式路徑為 temporal。`p` 是每個分析像素對應的長度；`dt` 是相鄰輸入影格的真實拍攝間隔。

| 指標 | 實作與使用的 mask | 單位及解讀限制 | 來源 |
|---|---|---|---|
| 面積 `area_px / area` | M 的前景像素數；`area=area_px×p²` | `_px` 是分析解析度上的像素面積數；`area` 才經尺度換算。是二維投影面積，不是體積／重量 | O:612–664；E:522–574 |
| 三層面積 | occupied、clean、envelope 分別計數及乘 p² | 必須標出是哪一層；envelope 填入網孔，通常更大 | E:1201–1246；temporal 指派 E:3117–3119 |
| 周長 `perimeter` | `measure.perimeter(M, neighborhood=8)×p` | 二值像素邊界估計；對孔洞、細枝、雜點、解析度敏感，不能保證等於實際組織外緣長度 | O:647–650；E:557–560 |
| Circularity | `4π×area_px/perimeter_px²` | 無單位的形狀緊密程度；網孔與分枝增加邊界會影響它，不能直接當生長好壞 | O:651–660；E:561–570 |
| Solidity | `regionprops` 的最大連通區面積／其凸包面積 | 無單位；**只取最大連通區**，但 area/perimeter 使用整張 M。多個斷開群體時統計對象不一致 | O:633–647、661；E:543–557、571 |
| Eccentricity | 最大連通區二階矩橢圓的離心率 | 無單位，表示分布偏長程度；不是分枝數，也不是對每條管線逐條擬合 | O:647、663；E:557、573 |
| Body skeleton 長度 | 對 M 骨架化後計數骨架像素，再乘 p | **目前是骨架像素數估計**；沒有對斜向步長乘 √2，也沒有建立量測用圖結構。團塊的中軸也會算入，不能稱為獨立偵測出的全部真實管長 | O:667–703；E:577–613 |
| Endpoints / branchpoints | 骨架像素的 8 鄰域：鄰居=1 計端點，≥3 計分叉像素 | 數的是像素，不是合併後的生物分叉節點；同一交會點可被算成數個分叉像素 | O:686–700；E:596–610 |
| 獨立 `vein_*` | 原圖在圓 ROI 與 body mask 限制內另做 HSV 或 ridge/Frangi，再骨架化；有時侵蝕 body 邊界 | 不是直接重命名 body skeleton。**目前入口關閉，不應宣稱現有主線已輸出獨立管網結果。** 分割中的 vesselness 補回仍開啟，但那是另一功能 | O:720 起、5961；E:630–761、8290；E:3183–3194 |
| 新增／退縮 | `new=M現在 & ~M前張`；`retracted=M前張 & ~M現在` | `_area_px` 是分析像素面積；未乘 p²。兩張 mask 差異同時含生物變化、分割誤差、震動及 ROI 截切 | O:977–1035；E:887–945 |
| Explore / retreat ratio | 新增／當張面積；退縮／前張面積 | 分母不同，兩者不能直接相減當成淨成長率；無單位 | O:1000–1007；E:910–917 |
| 淨面積變化與 `dA_dt` | `net_growth_px=當張像素數−前張像素數`；`dA_dt=(area現在−area前張)/dt` | 分別為分析像素面積與長度²/秒（dt 為秒時）。三層另有各自 dA/dt | O:957–975、1007、2729–2733；E:867–885、917、3204–3208 |
| `front_speed_mean / p95` | 所有新增像素到前張 mask 最近點的距離 ×p/dt，取平均與第95百分位 | 是新增區距離的速度代理值；**沒有追蹤同一個前緣點**，也不是單純最大前緣速度。寬新增區內部所有像素都參與平均 | O:1019–1023；E:929–933 |

當前最大連通區設定使多群體混用問題較少顯現；但新分割工具可能保留多個有效連通區，若接回舊量測，必須先決定全群體還是最大群體的統計政策。不能直接把新 mask 接入就宣稱所有形態指標定義不變。

`body_mask_confidence` 是依孔洞比例的啟發式分級（E:1221–1224），不是分割準確率。`apply_body_mask_confidence` 只產生部分 gated/trusted 欄（E:847–865）；天然多孔的網狀結構也可能被標 low，這不等於它是錯誤影像。

## 4. 取樣時間、解析度與單位

1. **10 秒是設定值，不是本次核對得到的實際拍攝時間。** 目前兩入口都要求使用真實拍攝間隔，而非 30 FPS 輸出播放速度（O:5895–5898；E:8169–8172）。若實際漏拍、影格重排、不同影像間隔，`frame_index×dt` 不能重建真實時間。正式 row 時間公式見 O:2738、E:3213；V2 分段保留來源 frame index，E:3635–3650、3934–3935。
2. 速度、面積變化率及 FFT 都依賴 dt。若每 k 張才取一張，必須使用 `k×原始拍攝間隔`；目前導數函式只做相鄰 row 差分／固定 dt（E:867–885），不讀每列實際時間差。預設正式影片流程逐張處理，這一點是日後抽樣或外部呼叫時的條件風險，不是宣稱預設已錯用 k。
3. `PIXEL_SIZE=1.0` 沒有提供 mm/px 校正證據。因此目前應報像素、像素²、像素/秒、像素²/秒（時間已確認時），不能標成 mm 或 mm²。若使用實測 p mm/px，對應 `area`、`perimeter`、`skeleton_length`、`front_speed` 才可換成實體單位。
4. 原版正式半解析度以 `effective_pixel_size=p/scale` 補償：O:2529–2540；V2 E:2926–2937。所以 `area` 約換回原解析度尺度，但 `_px` 類計數仍在縮小後座標。縮圖已失去的細管不會因乘回尺度恢復，周長與骨架也不保證尺度不變。
5. 第一個比較影格的新增／退縮計數預設為 0，而比例／速度為 NaN（E:892–904）。該 0 表示沒有前張可比較，不是已證實沒有變化。每個獨立片段都要保留此邊界語義。

## 5. 時間補回是否會蓋掉真實退縮？會，已有合成反例

V2 T:213–223 對前後 5 張 candidate 投票，只要達 4 票、候選連通區接觸當張前景附近，就可以補入。**它沒有要求該像素也存在於當張 candidate。** `exterior_only=True` 只排除當張填孔包圍區內的補回，不能判斷外側枝條是否真的縮回。

2026-11-18 以 32×40 的合成布林 mask 直接呼叫目前 T 函式，使用主入口的 window=5、min_hits=4、connect_radius=8、exterior_only=True、final_smooth_window=1；為隔離投票機制，關閉 motion_compensation：

- 基底團塊 100 px；外接枝條 28 px。
- 第 1、2、4、5 張有枝條，第 3 張的 mask 與 candidate 都沒有枝條，代表一個刻意設計的短暫真實消失。
- 第 3 張原面積 **100 → 輸出 128 px**；消失的 **28 px 全部被補回**；輸入 mask 陣列保持未修改。

這個反例證明演算法能抑制真實的短暫面積退縮；**沒有證明真實研究影片發生同樣大小的錯誤，也不代表所有收縮都會被蓋掉**。只要其他影格仍投票支持消失位置，就存在此種情形。光學亮度收縮與面積退縮也是不同量：前者即使輪廓不變仍可能存在；後者直接被 mask 補回影響。

原版的風險更廣：主入口 5/3 投票後還有 5/3 最後平滑（O:5939–5950），O:413–427 用多數 mask **覆寫**當張結果，可增加或移除當張像素。V2 關閉這一最後平滑；其可選平滑也重新併入當張 strict（T:236–245、253–256）。但兩者最後仍保留最大連通區，不能把此設計無條件描述為「保留任意多連通區的所有原始像素」。

時間窗使用未來影格，屬離線雙向修補。應將 raw spatial 與 repaired mask 分開保留；收縮／退縮分析不能把修補後曲線當作未經時間處理的直接觀察。

## 6. 震動目前處理到哪一步

### 原版

O:375–383 直接在同一像素位置對 candidate 投票，沒有影像配準；新增／退縮亦直接比較前後 mask（O:997–998）。培養皿偏移可能被當成外擴與退縮。

### V2 正式路徑

1. E:2998–3045 準備 item 時確實攜帶原圖及 dish geometry，所以正式路徑能進入 T 的配準，不是缺 geometry 的 identity 路徑。
2. 取培養皿 0.70–1.18 倍半徑環帶，CLAHE＋梯度後縮至最大尺寸 640，以 phase correlation 估平移；每批中間影格是 anchor。T:18–46、49–111。預設接受位移≤30 px、response≥0.35；**不估旋轉、不消除曝光變化**。
3. 有效估計者先平移候選與 strict 到共同 anchor 做投票，無效估計者留在原座標（T:189–207）。若同批部分成功、部分失敗，仍會共同投票，存在不同座標混用風險；不是「配準失敗就停止或把該張排除」。這是程式行為，尚未量化真實資料影響。
4. 投票後會平移回各張原座標（T:251–256）。E:3196–3202 隨即拿這些原座標 mask 計新增／退縮／速度，沒有再共同對齊。因此正式動態量仍可能受培養皿震動影響。不能將「motion_compensation_applied=True」解讀為所有動態指標已去震動。
5. Streaming 每個 core chunk 各自呼叫 temporal（E:3630–3705），anchor 隨批次改變；若要逐像素重現，需要保存 batch、來源區間及前後 halo。不能只保存 5/4 投票設定就宣稱所有不同批次完全一致。

### 新的獨立時間 QC

`segmentation_lab/temporal_qc.py:205` 的 `analyze_pair` 對原始單張 mask 做診斷，支援平移＋小旋轉；**不修改、不平滑、不補 mask**。配準不可信時 registered 指標留空。它比較 `前張有效 ROI ∩ 對齊後當張有效 ROI ∩ 影像有效範圍`（同檔 254–272），避免裁切邊界被誤算成生物退縮。原始面積與 raw 比較另外保留，不能以共同視野面積取代完整當張面積。

2026-11-18 回歸測試：256² 紋理環帶，ROI 中心(128,128)、半徑94.72、margin5；靜態黃色矩形穿越固定 ROI 邊界，整張向右平移4 px。修正後 registered IoU=1、new=0、retracted=0，輸入 mask 不變。這是合成條件的邏輯驗證，不是所有真實震動皆可成功消除。無共同有效 ROI 亦不產假量測。CLI 會確認 seek 與實際 decoded index，記錄程式 hash、有效參數及 acquisition interval；未知時間保持空值，不使用播放 FPS 代替。

## 7. 「收縮分析」實際是光學代理量

### 原版

`roi_mean_intensity` 是固定矩形內**所有灰階像素**的平均，沒有用黏菌 mask 限制（O:1038–1051）。`analyze_oscillation` 對平均亮度去趨勢後做 FFT 與峰值計數（O:1456–1503）；缺值會直接線性補齊所有 gap，沒有 gap 長度限制（O:1472–1475），也沒有指定生理週期範圍。目前 `ROI=None`，這項不啟動。

### V2 固定 ROI 強度法

預設灰階來源、clean layer、ROI 中至少50個有效像素（E:8188–8195）。`contraction_roi_intensity_stats` 只平均 ROI∩指定 mask 內像素；不足數量即回 NaN（E:1057–1102）。目前 temporal 路徑 clean=temporal，因此**時間補回也會改變納入亮度平均的像素集合**（E:3246–3265）。即使每個組織像素本身亮度不變，選入區域改變也能讓平均值變化；亮度訊號不能自動歸因於細胞收縮。

處理為：最多3張短 gap 插值 → 取最長連續有效片段 → 去線性趨勢 → FFT → 在30–600秒範圍選最大功率頻率（E:1390–1494、8191–8195）。summary 另檢查至少20張有效資料、有效比例≥0.8、正振幅、估計至少2個週期（E:1636–1654）。品質標記也會指出資料長度是否不足涵蓋兩個最大允許週期（E:1467–1475）。

輸出週期為秒、頻率為 Hz 的前提是 dt 確為秒且均勻。振幅是灰階數值（DN），欄名 `amplitude_peak_to_peak` 實際算去趨勢後 **P95−P5**，不是 max−min（E:1485–1488）。找出範圍內最大頻峰不等於證明存在顯著生物週期；此實作沒有噪音虛無假設／頻峰顯著性檢定。程式自己也明示是光學代理，非收縮力或直接原生質流量（E:979–983）。

### V2 追蹤法

樣本收縮流程另有模板追蹤與局部光流。灰階先 CLAHE（E:6746–6750）；追蹤位置、局部 flow、影格差形成候選訊號（E:6970–7003、7039–7044）。它不依賴 body mask 逐像素限制，選定矩形內仍可含背景。

- 位移是像素；local flow magnitude 是相鄰畫面位移量，該函式沒有乘 p／除 dt；影格差是灰階數值。不能把這些欄全部解讀為 mm/s 或力。
- 追蹤分數不足時凍結方框，仍將位置／亮度／flow 等數字加入序列（E:6975–6996）；`n_valid` 主要按訊號是否有限值計算（E:7056–7058），不是逐張追蹤成功數。
- `best_signal` 用不同單位訊號的原始振幅比較，且不先要求 valid（E:7101–7116）。因此「best」不能解讀為最可信的生理訊號。
- 沒有先扣掉培養皿整體運動，位置／光流含相機或培養皿震動。來源已標為 tracker-like optical proxy（E:7095–7098）。它與上節只用於 temporal 投票的平移補償不是同一條管線。

## 8. 後續整合應保持的界線

1. 量測主 mask 定義為當張可見 occupied，食物排除、孔洞與多連通區政策明示；envelope、修補 mask 與原結果分別保存。
2. 震動診斷使用可信配準與共同有效 ROI；配準失敗或無重疊留空，不以 identity 或 0 包裝成成功。不要以 IoU 提升直接當作分割準確率提升。
3. 真實退縮與週期分析保留未時間補回的序列，修補版本只作敏感度比較；先建立人工確認樣本再判斷是否能作研究終點。
4. 明確區分 body skeleton 與獨立 vein network；若要真實管長／分叉數，另定義斜向長度、交會點合併及雜枝政策，再建立驗證。
5. 真實 acquisition 時間與實測尺度是速度、週期、實體面積的必要來源；播放 FPS 和 pixel_size=1 不能替代。

## 9. 本次確認範圍與來源指紋

- 重新讀取目前入口、模組綁定、指標公式、時間投票、固定 ROI 與 tracker 路徑。
- 直接呼叫 V2 temporal 的32×40合成反例，中央100→128 px；沒有讀入或改寫研究影片。
- 獨立 temporal QC 的12項合成測試通過（6.06 s），包含剛體移動、真實新增、配準失敗、共同 ROI 邊界、輸出 mask 不變、未知時間、seek／frame identity 失敗。此測試不驗證舊 V2 的生物準確率。
- 主程式來源 SHA-256（本次查核後再次讀取）：

```text
plasmodium_brightfield_analysis.py
0ba203a6524921cc05d911a11aff65af4deef6d319b5cfc53d0bae02d3664c72
plasmodium_analysis_v2/plasmodium_v2/engine.py
d60c2c9554e43b72c8ceeefed1f7f2802422b7ccf2294b2574cfbfa6c8ada5d0
plasmodium_analysis_v2/plasmodium_v2/temporal.py
bf5d43a0e7b05b938835e130f21cb09d0fd8a013fcb1c3afb9f2f1f770bda9ad
plasmodium_analysis_v2/user_config.py
83f036ecdf61a24134fa336ef94b2d2118ca7c513f2f8b72035f51c55d3ed006
```

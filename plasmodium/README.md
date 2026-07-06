# 黏菌明視野影像分析：程式量化內容與文獻解讀

> 更新：2026-07-06  
> 本文件只說明程式計算什麼、影像可如何解讀，以及文獻依據。  
> **30 張人工 mask 尚未精確完成，正式結果也尚未重跑，因此不報 Dice、IoU、準確率或生物效應。**

[開啟圖文 HTML 版](PLASMODIUM_ANALYSIS_REPORT.html)

## 這支程式在做什麼

`plasmodium_brightfield_analysis.py` 將 *Physarum polycephalum*（多頭絨泡菌）明視野縮時影片逐格轉成黑白遮罩，再把肉眼看到的擴張、縮回、變形與分枝轉成可畫圖、可比較的數字。

程式目前可回答：

- 每一格影像中有多少二維範圍被判定為黏菌？
- 輪廓是緊密、圓整，還是凹凸、分枝化？
- 相鄰影格間，哪些區域新出現、哪些區域縮回？
- 中心線網路有多長、多少端點與分叉候選？
- 影像是否失焦、遮罩是否碰邊或可能不可靠？
- 淡色分枝能否利用前後影格的一致性補回？

它不能單獨回答「黏菌是否健康」、「某處理是否造成變化」或「微重力與 1g 是否不同」。這些結論仍需要對照組、獨立培養、生物重複、精確標註與統計。

## 目前設定與資料狀態

| 項目 | 目前設定 | 說明 |
|---|---|---|
| 輸入 | `output2.mp4` | 來源縮時影片 |
| 真實取像間隔 | `FRAME_INTERVAL = 10.0 s` | 時間與變化率依每 10 秒一格計算，不使用輸出影片播放 FPS |
| 分割模式 | `hsv_circle` | 依黃色 HSV 特徵找黏菌 |
| 培養皿範圍 | 圓心 `(1280, 920)`、半徑 `680 px` | 排除皿外背景 |
| 目前執行模式 | `test_front_middle_rear` | 只輸出前、中、後測試片，不是正式全片結果 |
| 像素尺寸 | 尚未實體校正 | 長度與面積應寫 px、px²，不可直接寫 mm、mm² |
| 人工 mask | 尚在精修 | 目前不使用草稿 mask 宣稱準確率 |
| vein/ridge | `VEIN_EVERY_N_FRAMES = 0` | 目前正式設定停用獨立脈管分析 |

資料夾內既有 CSV、圖與人工比較檔均視為開發產物；需等精確 mask 與正式設定完成後重跑。

## 影像處理流程

1. **讀取影格與換算時間**：第 `i` 格時間為 `i × 10 秒`。
2. **限制培養皿範圍**：只在指定圓內分析，降低皿外背景干擾。
3. **HSV 黃色分割**：H 表色相、S 表顏色濃淡、V 表亮度；用三者找黃色黏菌。
4. **連通與形態清理**：移除小色點、保留主要相連本體，有限度修補小孔。
5. **時間補償候選**：淡色像素必須在多格持續、靠近本體、避開皿緣才可補入。
6. **量化與品管**：輸出逐格 CSV、遮罩疊圖、摘要圖和可疑格清單。

核心是「**嚴格顏色種子 + 受限擴張 + 時間一致性**」。全面降低 HSV 門檻會同時增加皿緣、反光和背景誤抓，因此不作為預設方法。

## 三層黏菌遮罩

| 遮罩 | 主要欄位 | 定義 | 用途 |
|---|---|---|---|
| 可見佔據區 | `occupied_area_px` | HSV 實際看見的網路，保留內部空隙 | 描述可見黏菌網路 |
| 清理後本體 | `clean_body_area_px` | 只修補有限小孔後的本體 | 目前形態分析主遮罩 |
| 外包絡 | `envelope_area_px` | 將整個外輪廓包起來 | 看涵蓋範圍與孔洞；不是實際黏菌量 |

`filled_small_hole_area_px` 記錄程式補了多少小孔；`remaining_hole_fraction` 記錄清理後仍有多少內部空區。真實網路空隙與漏抓都可能造成孔洞，因此必須搭配原圖判讀。

## 程式量化的指標

### 1. 影像與遮罩品質

| 欄位 | 如何計算／用途 | 正確解讀 |
|---|---|---|
| `focus_score` | 灰階影像的 Laplacian variance | 同一設備內相對清晰度；不是跨相機絕對標準 |
| `mask_area_fraction` | 遮罩面積 ÷ 分析畫面面積 | 畫面被遮罩覆蓋的比例 |
| `touch_roi_border` | 遮罩是否碰分析邊界 | `True` 時輪廓可能被裁掉 |
| `body_mask_confidence` | 依遮罩與孔洞規則分級 | 內部 QC 等級，不是準確率 |
| `remaining_hole_fraction` | 未填內孔 ÷ 外包絡 | 網路空隙或漏抓的警示，兩者需人工區分 |

### 2. 覆蓋面積與大小

| 欄位 | 定義 | 可以代表 | 不可直接代表 |
|---|---|---|---|
| `area_px` | 遮罩像素總數 | 二維可見覆蓋範圍 | 體積或生物量 |
| `area` | `area_px × pixel_size²` | 校正後面積 | 未校正時的 mm² |
| `bbox_area_px` | 最小水平外接矩形面積 | X/Y 方向占用範圍 | 真實黏菌面積 |
| `occupied_area_px` | 可見網路面積 | 含真實空隙的可見範圍 | 外輪廓內全部區域 |
| `envelope_area_px` | 外包絡面積 | 外形涵蓋範圍 | 實際組織面積 |

Tran 等（2015）指出黏菌擴張輪廓不規則，影像像素面積比尺量半徑更適合量固體培養基上的擴張；該研究也比較了面積與乾重。這支持本程式量二維面積，但本實驗沒有自己的乾重校正，因此只能稱「可見覆蓋面積」。[[論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5611581/)]

### 3. 輪廓與形狀

| 欄位 | 公式／定義 | 影像意義 |
|---|---|---|
| `perimeter_px` | 外輪廓長度 | 同時受大小和邊界曲折度影響 |
| `circularity` | `4πA/P²` | 接近 1 越圓；下降常見於伸出偽足、分枝或遮罩破碎 |
| `solidity` | `A/convex hull area` | 接近 1 越緊密；下降表示凹陷、指狀突出或分散增加 |
| `eccentricity` | 等效橢圓偏心率 | 近 0 較圓，近 1 較細長 |

這些不是健康分數。形狀改變可能來自正常覓食、環境刺激、營養狀態，也可能只是失焦或分割錯誤。

### 4. 探索、縮回與淨面積變化

```text
new_region       = M_t AND NOT M_(t-1)
retracted_region = M_(t-1) AND NOT M_t
explore_ratio    = new_area / current_area
retreat_ratio    = retracted_area / previous_area
net_growth_px    = current_area - previous_area
dA_dt            = net_growth / 10 seconds
```

| 欄位 | 近似意義 | 不能直接代表 |
|---|---|---|
| `new_area_px` | 本格新覆蓋區 | 新生成的生物量 |
| `retracted_area_px` | 上格有、本格消失區 | 細胞死亡 |
| `explore_ratio` | 相對目前大小的新探索比例 | 單一方向移動速度 |
| `retreat_ratio` | 相對上一格大小的縮回比例 | 死亡率 |
| `net_growth_px` | 新增與縮回抵消後的淨差 | 真實質量增加 |
| `dA_dt` | 每秒二維淨面積變化 | 細胞質流速 |

### 5. 前緣移動

`front_speed_mean` 與 `front_speed_p95` 由新出現區域離舊遮罩邊界的距離除以 10 秒估算。它描述的是**影像平面上的遮罩前緣速度**，不是管內細胞質流速。相機位移、未配準或遮罩閃爍都會造成假速度。

### 6. 骨架、端點與分叉

| 欄位 | 計算 | 可能意義 | 主要風險 |
|---|---|---|---|
| `skeleton_length_px` | 一像素中心線總長 | 網路延伸程度 | 漏一條淡枝就會低估 |
| `n_endpoints` | 中心線 8 鄰域只有 1 個鄰居 | 末端／探索尖端候選 | 小毛刺會製造假端點 |
| `n_branchpoints` | 8 鄰域至少 3 個鄰居 | 分叉候選 | 像素化可能重複計數 |

黏菌會持續增強、裁剪和重組管狀網路，因此量中心線與分叉有生物學理由；但現階段只能作探索性描述。[[Patino-Ramirez et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7810851/)] 管狀結構還需用 clDice 等中心線／拓撲指標驗證，不能只看面積重疊。[[Shit et al., 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation_CVPR_2021_paper.html)]

### 7. 脈管與亮度振盪

程式保留 `vein_area_px`、`vein_length_px`、脈管端點／分叉及 ROI 亮度 FFT 功能，但目前 `VEIN_EVERY_N_FRAMES=0`，獨立脈管分析未啟用。

亮度 FFT 可輸出 `osc_dominant_freq`、`osc_period`、`osc_n_peaks`。亮度週期可能與厚度和收縮波有關，但曝光、照明漂移、失焦和遮罩改變也會產生週期。因此目前不能宣稱量到真實管徑、管內流量、shuttle streaming 或收縮頻率。

## 依黏菌習性，影像可以怎麼說明

以下是文獻支持的**候選解釋**，不是看圖即可確定的結論。

### 扇形、密集的前緣

*Physarum* 的探索前緣常呈較密集的扇形，前方伸出偽足；後方則逐漸整理成管狀網路。因此影像若見前端寬厚、後方細管化，可描述為「符合探索前緣與後方輸送網路的典型空間分化」。不能只憑外觀判定營養方向，仍需知道食物與刺激位置。[[Gerber et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35195068/)]

### 輪廓由圓整轉為指狀或多分枝

在均質環境中，黏菌可先向各方向伸展，之後出現偽足並進入分枝化探索。影像上可能表現為 circularity、solidity 下降，端點增加。這可稱為「由緊密形態轉向較分枝的探索形態」，但也必須排除遮罩毛刺。[[Patino-Ramirez et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6817824/)]

### 一端新增、另一端縮回

黏菌遷移與重塑時不一定讓總面積明顯增加；它可能前方延伸、後方回收。此時 `new_area_px` 與 `retracted_area_px` 可同時偏高，而 `net_growth_px` 接近零。這較符合「位置轉移／重塑」而非「停止活動」。

### 管路逐漸增粗、細枝消失

黏菌網路會依運輸需求強化部分路徑、裁剪效率較低的支路。影像若見部分主幹變粗、細枝減少，可提出「網路正在重新配置或集中輸送路徑」的假說。要確認仍需管徑量測、流動資料與重複樣本，不能只用二值骨架。[[Tero et al., 2010](https://doi.org/10.1126/science.1177894)] [[Kramar & Alim, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7958412/)]

### 探索變慢、形狀更緊密

既有研究顯示均勻營養環境或不利環境都可能讓擴張變慢，但原因不同：前者可能停留代謝營養，後者可能降低暴露並產生保護性黏液。影像若只見面積成長變慢，不能直接判斷是「營養好」還是「受到壓力」；需要 substrate、mucus、形狀和行為的共同證據。[[Patino-Ramirez et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6817824/)]

### 規律的亮暗或粗細變化

黏菌管路會週期性收縮與放鬆，驅動細胞質往返流動（shuttle streaming），並參與刺激訊號的長距離傳播。若固定 ROI 出現規律亮暗變化，可列為收縮振盪候選；但必須先排除相機自動曝光、光源波動和焦點漂移，最好再以管徑或流速量測驗證。[[Alim et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441820/)]

### 顏色變淡或局部消失

影像變淡可能是黏菌變薄、管路被裁剪、色素／厚度改變、失焦或照明變化；也可能只是 HSV 漏抓。不能直接解讀成死亡。時間補償的用途是辨認「在前後多格持續存在、但單格顏色不夠強」的候選分枝，最後仍需人工精確遮罩確認。

## 為什麼必須合併多個指標

| 影像組合 | 較合理的描述 | 還需排除 |
|---|---|---|
| area 上升、explore 高、retreat 低 | 可見範圍淨擴張 | 顏色變深造成遮罩增加 |
| area 近似不變、explore 與 retreat 都高 | 邊探索邊縮回，可能遷移或重塑 | 相機位移、遮罩閃爍 |
| circularity/solidity 下降、骨架增加 | 形態較分枝、較不緊密 | 毛刺與漏抓 |
| 主幹增粗、細枝減少 | 可能為路徑強化與裁剪 | 二值遮罩無法可靠量管徑 |
| focus 下降且 area 下降 | 可能失焦導致漏抓 | 真實回縮 |
| 規律亮暗變化 | 收縮振盪候選 | 曝光、照明、焦點漂移 |

Rosina 與 Grube（2024）同時評估生長、探索、管徑、收縮與網路複雜度，正說明單一面積曲線不足以代表黏菌的全部結構與功能反應。[[論文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11528663/)]

## 人工 mask 完成後的正式驗證

| 指標 | 回答的問題 |
|---|---|
| Dice、IoU | 整體區域重疊多少 |
| Precision | 電腦圈出的區域有多少是真的；低值表示背景誤抓多 |
| Recall | 人工黏菌有多少被找到；低值表示漏抓多 |
| signed/absolute area error | 面積是否系統性高估或低估 |
| Boundary F1、HD95 | 外輪廓位置是否準確 |
| clDice | 細枝中心線與連通性是否保留 |
| skeleton/endpoint/branchpoint error | 網路幾何與拓撲是否可靠 |
| temporal change error | 新增、縮回與淨變化是否可信 |

Maier-Hein 等（2024）強調指標要配合研究問題：研究覆蓋面積應重視 area error；研究細枝網路就必須加入邊界與拓撲指標。[[Metrics Reloaded](https://www.nature.com/articles/s41592-023-02151-z)]

正式驗證需先完成精確標註，再將開發與最終測試資料分開、凍結程式和參數、分前中後時段報告，並以獨立培養而非相鄰影格作統計單位。完整流程見 [`publication_validation_system/VALIDATION_WORKFLOW_ZH.md`](publication_validation_system/VALIDATION_WORKFLOW_ZH.md)。

## 主要輸出檔案

| 檔案 | 用途 | 現階段定位 |
|---|---|---|
| `run_mode_indicator.csv` | 記錄本次跑 formal 或 test | 執行紀錄 |
| `plasmodium_metrics.csv` | 每格形態與動態值 | 等正式設定重跑 |
| `plasmodium_summary.png` | 時間序列摘要 | 等正式設定重跑 |
| `plasmodium_overlay.mp4` | 原圖與遮罩疊合 | 開發期目視 QC |
| `plasmodium_qc_flags.csv` | 失焦、跳動、碰邊等清單 | 人工回看清單，不是自動刪除名單 |
| `manual_validation_*.csv` | 自動與人工 mask 比較 | 等人工 mask 精修後重跑 |
| `validation_clips/` | 前中後段視覺測試 | 穩定性檢查，不是正式結果 |

## 參考文獻

1. Tran HTM, et al. GIS-based image analysis for assessing growth of *Physarum polycephalum*. 2015. [doi:10.1186/s40694-015-0017-z](https://doi.org/10.1186/s40694-015-0017-z).
2. Patino-Ramirez F, et al. Substrate composition directs slime molds behavior. 2019. [doi:10.1038/s41598-019-50872-z](https://doi.org/10.1038/s41598-019-50872-z).
3. Patino-Ramirez F, Arson C, Dussutour A. Substrate and cell fusion influence on slime mold network dynamics. 2021. [doi:10.1038/s41598-020-80320-2](https://doi.org/10.1038/s41598-020-80320-2).
4. Tero A, et al. Rules for biologically inspired adaptive network design. 2010. [doi:10.1126/science.1177894](https://doi.org/10.1126/science.1177894).
5. Alim K, et al. Mechanism of signal propagation in *Physarum polycephalum*. 2017. [doi:10.1073/pnas.1618114114](https://doi.org/10.1073/pnas.1618114114).
6. Kramar M, Alim K. Encoding memory in tube diameter hierarchy of living flow network. 2021. [doi:10.1073/pnas.2007815118](https://doi.org/10.1073/pnas.2007815118).
7. Rosina P, Grube M. Novel image-analytic approach reveals new insights in fine-tuning of slime mould network adaptation. 2024. [doi:10.1098/rsos.240950](https://doi.org/10.1098/rsos.240950).
8. Shit S, et al. clDice: topology-preserving tubular structure segmentation. 2021. [Open access](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation_CVPR_2021_paper.html).
9. Maier-Hein L, et al. Metrics Reloaded. 2024. [doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z).

## 對外摘要

> 目前已完成黏菌縮時影像的自動分割與量化框架。程式可計算二維覆蓋面積、輪廓形狀、相鄰影格的探索與縮回、中心線及影像品質，並測試利用時間一致性補回淡色分枝。文獻支持這些影像特徵可用來描述黏菌的探索、遷移與網路重塑，但人工 mask 與正式重跑尚未完成，因此現階段只說明方法與可能的生物學意義，不報演算法準確率或實驗效應。

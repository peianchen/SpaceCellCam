import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "file:///C:/Users/chen8/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const assetsDir = process.argv[2];
const outputPath = process.argv[3];
const qaDir = process.argv[4];
await fs.mkdir(qaDir, { recursive: true });

const C = {
  navy: "#203864",
  blue: "#4472C4",
  orange: "#ED7D31",
  green: "#70AD47",
  ink: "#243247",
  muted: "#667085",
  line: "#D8E0EC",
  soft: "#F4F7FB",
  white: "#FFFFFF",
};
const FONT = "Microsoft JhengHei";
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

async function bytes(name) {
  return new Uint8Array(await fs.readFile(`${assetsDir}/${name}`));
}

function addText(slide, name, value, x, y, w, h, style = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = value;
  s.text.style = {
    typeface: FONT,
    fontSize: 18,
    color: C.ink,
    verticalAlignment: "middle",
    ...style,
  };
  return s;
}

function addBox(slide, name, x, y, w, h, fill = C.white, line = C.line, radius = "rounded-xl") {
  return slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1.2 },
    borderRadius: radius,
  });
}

async function addImage(slide, name, assetName, position, options = {}) {
  const ext = assetName.toLowerCase();
  const contentType = ext.endsWith(".jpeg") || ext.endsWith(".jpg") ? "image/jpeg" : "image/png";
  return slide.images.add({
    blob: await bytes(assetName),
    contentType,
    alt: options.alt ?? name,
    fit: options.fit ?? "cover",
    position,
    crop: options.crop,
    geometry: options.geometry ?? "roundRect",
    borderRadius: options.borderRadius ?? "rounded-xl",
  });
}

function addHeader(slide, title, section, number) {
  slide.background.fill = "#F8FAFC";
  slide.shapes.add({
    geometry: "chevron",
    name: `header-orange-${number}`,
    position: { left: 28, top: 26, width: 48, height: 48 },
    fill: C.orange,
    line: { style: "solid", fill: "none", width: 0 },
  });
  slide.shapes.add({
    geometry: "chevron",
    name: `header-blue-${number}`,
    position: { left: 61, top: 26, width: 48, height: 48 },
    fill: C.blue,
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(slide, `slide-title-${number}`, title, 130, 22, 1000, 60, {
    fontSize: 35,
    bold: true,
    color: C.navy,
  });
  addText(slide, `section-${number}`, section.toUpperCase(), 1010, 84, 210, 22, {
    fontSize: 11,
    bold: true,
    color: C.muted,
    alignment: "right",
  });
  slide.shapes.add({
    geometry: "rect",
    name: `header-rule-${number}`,
    position: { left: 56, top: 110, width: 1168, height: 2 },
    fill: C.line,
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(slide, `page-${number}`, `${String(number).padStart(2, "0")} / 07`, 1120, 681, 100, 18, {
    fontSize: 11,
    color: C.muted,
    alignment: "right",
  });
}

function setNotes(slide, value) {
  slide.speakerNotes.textFrame.setText(value);
  slide.speakerNotes.setVisible(true);
}

// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";
  slide.shapes.add({
    geometry: "rect",
    name: "title-accent-orange",
    position: { left: 62, top: 125, width: 9, height: 150 },
    fill: C.orange,
    line: { style: "solid", fill: "none", width: 0 },
  });
  slide.shapes.add({
    geometry: "rect",
    name: "title-accent-blue",
    position: { left: 78, top: 125, width: 9, height: 150 },
    fill: C.blue,
    line: { style: "solid", fill: "none", width: 0 },
  });
  addText(slide, "title-eyebrow", "SPACE BIOMEDICAL RESEARCH", 110, 98, 520, 32, {
    fontSize: 15,
    bold: true,
    color: C.blue,
  });
  addText(slide, "title-main", "進度報告", 110, 150, 520, 92, {
    fontSize: 58,
    bold: true,
    color: C.navy,
  });
  addText(slide, "title-sub", "穿戴式感測與黏菌影像分析", 110, 250, 540, 62, {
    fontSize: 28,
    bold: true,
    color: C.ink,
  });
  addText(slide, "title-date", "2026.07.03", 110, 350, 260, 40, {
    fontSize: 22,
    color: C.muted,
  });
  addText(slide, "title-scope", "低功耗韌體、週期影像擷取、HSV 分割與驗證規劃", 110, 410, 520, 78, {
    fontSize: 18,
    color: C.muted,
  });
  await addImage(slide, "title-illustration", "title-illustration-crop.png", { left: 650, top: 42, width: 580, height: 630 }, {
    fit: "contain",
    alt: "Research data presentation illustration from the source deck",
  });
  setNotes(slide,
    "老師好，今天的進度分成兩部分。第一部分是穿戴式感測裝置的低功耗與 BLE 連線；第二部分是黏菌長時間影像的擷取與量化。後半段會說明 HSV 分割不是單純調門檻，而是先建立高可信種子，再用連通條件補回淡色區域。最後我會把目前能展示的結果，和仍需完成的驗證工作分開說明。"
  );
}

// Slide 2
{
  const slide = presentation.slides.add();
  addHeader(slide, "穿戴裝置已完成主要低功耗路徑", "BLE wearable", 2);
  await addImage(slide, "low-power-bench", "image2.png", { left: 56, top: 142, width: 500, height: 355 }, {
    crop: { left: 0.02, top: 0.02, right: 0.12, bottom: 0.02 },
    alt: "Low-power bench measurement showing about 79 microamps in the displayed state",
  });
  addText(slide, "bench-caption", "待機狀態照片讀值約 79 µA", 72, 508, 468, 32, {
    fontSize: 18,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  addText(slide, "wearable-done-heading", "本輪完成", 600, 140, 320, 38, {
    fontSize: 24,
    bold: true,
    color: C.navy,
  });
  const items = [
    ["60 秒取樣", "降低感測與喚醒頻率"],
    ["關閉 UART / log", "Release 量測不開 serial console"],
    ["I²C suspend", "讀完 MLX 後停用 peripheral"],
    ["BLE interval + latency", "降低連線期間的喚醒次數"],
    ["MLX 電源閘控", "由 GPIO 控制 MOSFET 斷電"],
  ];
  items.forEach((item, idx) => {
    const y = 190 + idx * 66;
    slide.shapes.add({
      geometry: "ellipse",
      name: `wearable-dot-${idx}`,
      position: { left: 600, top: y + 10, width: 12, height: 12 },
      fill: idx === 4 ? C.orange : C.blue,
      line: { style: "solid", fill: "none", width: 0 },
    });
    addText(slide, `wearable-title-${idx}`, item[0], 624, y, 330, 26, {
      fontSize: 18,
      bold: true,
      color: C.ink,
    });
    addText(slide, `wearable-detail-${idx}`, item[1], 624, y + 27, 330, 26, {
      fontSize: 14,
      color: C.muted,
    });
  });
  await addImage(slide, "wearable-app", "image3.png", { left: 1000, top: 138, width: 230, height: 500 }, {
    fit: "contain",
    alt: "BLE monitoring app showing the connected MLX sensor",
  });
  addBox(slide, "wearable-result", 56, 565, 900, 78, "#FFF4EA", "#F3B37A");
  addText(slide, "wearable-result-text", "最大單項改善：MLX 斷電約降低 1 mA\n照片讀值代表特定待機狀態，完整續航仍需用長時間平均電流計算。", 80, 575, 850, 58, {
    fontSize: 17,
    bold: true,
    color: C.ink,
  });
  setNotes(slide,
    "這一頁先說穿戴裝置。低功耗不是只靠一個設定，而是把每個會持續耗電的路徑逐一關閉。取樣改成每六十秒一次，Release 版本關閉 UART 與 log，讀完 MLX 後停用 I²C，BLE 也調整 connection interval 和 latency。影響最大的是 MLX 的電源閘控，原始測試約可降低一毫安。左圖顯示某個待機狀態約七十九微安，但這不是整個工作週期的平均值，因此續航還要用長時間電流紀錄計算。右圖則確認 BLE App 已能找到並顯示 MLX 感測器。"
  );
}

// Slide 3
{
  const slide = presentation.slides.add();
  addHeader(slide, "影像擷取已週期化，焦距仍是辨識品質關鍵", "camera pipeline", 3);
  addText(slide, "camera-flow-heading", "每一個拍攝週期", 56, 140, 340, 36, {
    fontSize: 24,
    bold: true,
    color: C.navy,
  });
  const flow = [
    ["1", "喚醒相機", "Software Standby → Active"],
    ["2", "開燈與對焦", "等待曝光與 AF 穩定"],
    ["3", "擷取並存檔", "JPEG → PSRAM → SD"],
    ["4", "關燈待機", "回到 Software Standby"],
  ];
  const nodes = flow.map((item, idx) => {
    const y = 190 + idx * 104;
    const node = addBox(slide, `camera-step-${idx + 1}`, 56, y, 340, 76, idx === 1 ? "#FFF4EA" : C.white, idx === 1 ? "#F3B37A" : C.line);
    addText(slide, `camera-number-${idx + 1}`, item[0], 72, y + 16, 44, 44, {
      fontSize: 20,
      bold: true,
      color: C.white,
      alignment: "center",
    });
    slide.shapes.add({
      geometry: "ellipse",
      name: `camera-number-bg-${idx + 1}`,
      position: { left: 72, top: y + 16, width: 44, height: 44 },
      fill: idx === 1 ? C.orange : C.navy,
      line: { style: "solid", fill: "none", width: 0 },
    });
    addText(slide, `camera-number-top-${idx + 1}`, item[0], 72, y + 16, 44, 44, {
      fontSize: 20,
      bold: true,
      color: C.white,
      alignment: "center",
    });
    addText(slide, `camera-step-title-${idx + 1}`, item[1], 132, y + 8, 240, 30, {
      fontSize: 19,
      bold: true,
      color: C.navy,
    });
    addText(slide, `camera-step-detail-${idx + 1}`, item[2], 132, y + 38, 240, 26, {
      fontSize: 14,
      color: C.muted,
    });
    return node;
  });
  for (let i = 0; i < nodes.length - 1; i += 1) {
    slide.shapes.connect(nodes[i + 1], nodes[i], {
      kind: "straight",
      fromSide: "top",
      toSide: "bottom",
      line: { style: "solid", fill: "#8EA0B8", width: 2 },
      head: { type: "triangle", width: "sm", length: "sm" },
    });
  }
  addText(slide, "focus-before-label", "焦距調整前", 446, 140, 340, 34, {
    fontSize: 20,
    bold: true,
    color: C.muted,
    alignment: "center",
  });
  addText(slide, "focus-after-label", "焦距調整後", 840, 140, 340, 34, {
    fontSize: 20,
    bold: true,
    color: C.green,
    alignment: "center",
  });
  await addImage(slide, "focus-before", "image3.jpeg", { left: 440, top: 180, width: 350, height: 350 }, {
    crop: { left: 0.06, top: 0.05, right: 0.06, bottom: 0.06 },
    alt: "Plasmodium image before lens RPM adjustment",
  });
  await addImage(slide, "focus-after", "image4.jpeg", { left: 830, top: 180, width: 350, height: 350 }, {
    crop: { left: 0.06, top: 0.05, right: 0.06, bottom: 0.06 },
    alt: "Plasmodium image after lens RPM adjustment",
  });
  addBox(slide, "focus-takeaway", 440, 565, 740, 72, "#EAF2FB", "#BFD0E8");
  addText(slide, "focus-takeaway-text", "焦距會直接改變細枝可見度；正式長時間拍攝前必須先鎖定鏡頭位置。", 465, 577, 690, 48, {
    fontSize: 19,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  setNotes(slide,
    "相機端目前已經形成固定週期。開始時從 Software Standby 喚醒 OV5640，開啟 LED，等待曝光和自動對焦穩定，再把 JPEG 先放到 PSRAM，接著寫入 SD，最後關燈並回到待機。右側是鏡頭 RPM 調整前後的比較。這裡要強調的不是單張影像好不好看，而是焦距會直接影響淡色細枝是否可見；如果拍攝期間焦距漂移，後續分割面積就會被誤當成生長變化。因此正式序列要先固定鏡頭位置。"
  );
}

// Slide 4
{
  const slide = presentation.slides.add();
  addHeader(slide, "面積能顯示長期趨勢，但瞬時變化仍含高頻波動", "signal extraction", 4);
  await addImage(slide, "segmented-frame", "image7.png", { left: 56, top: 145, width: 510, height: 470 }, {
    crop: { left: 0.02, top: 0.02, right: 0.02, bottom: 0.02 },
    alt: "Segmented frame with body mask and internal network overlay",
  });
  addText(slide, "area-label", "遮罩面積", 610, 130, 610, 28, {
    fontSize: 18,
    bold: true,
    color: C.green,
    alignment: "center",
  });
  await addImage(slide, "area-chart", "area-chart-crop.png", { left: 610, top: 158, width: 610, height: 210 }, {
    fit: "contain",
    alt: "Area trend extracted from the segmented sequence",
  });
  addText(slide, "derivative-label", "面積變化率 dA/dt", 610, 370, 610, 28, {
    fontSize: 18,
    bold: true,
    color: C.blue,
    alignment: "center",
  });
  await addImage(slide, "derivative-chart", "derivative-chart-crop.png", { left: 610, top: 398, width: 610, height: 210 }, {
    fit: "contain",
    alt: "Area derivative showing increasing high-frequency fluctuations",
  });
  addBox(slide, "signal-takeaway", 56, 626, 1164, 46, "#203864", "#203864");
  addText(slide, "signal-takeaway-text", "目前可穩定取得長期面積趨勢；後段 dA/dt 波動放大，需再分離真實形變與影像擾動。", 80, 630, 1116, 38, {
    fontSize: 17,
    bold: true,
    color: C.white,
    alignment: "center",
  });
  setNotes(slide,
    "分割遮罩之後，可以把每一幀轉成面積時間序列。上方綠線顯示長時間尺度下，遮罩面積整體上升，因此這個指標有能力描述生長趨勢。但下方的 dA/dt 在後段出現很大的正負尖峰，代表瞬時變化對焦距、亮度、邊界抖動或分割誤差非常敏感。這一頁不把尖峰直接解釋成生物行為；目前能確認的是長期面積可量化，而高頻成分還需要平滑、品質控管與人工標註驗證。"
  );
}

// Slide 5
{
  const slide = presentation.slides.add();
  addHeader(slide, "HSV 先找高可信種子，再用連通限制補回淡色區域", "segmentation algorithm", 5);
  await addImage(slide, "hsv-comparison", "image5.jpeg", { left: 56, top: 140, width: 470, height: 520 }, {
    fit: "contain",
    alt: "Front, middle, and rear original images with segmentation overlays",
  });
  const hsvItems = [
    ["H", "色相", "鎖定黃色", C.orange, "#FFF3E8"],
    ["S", "飽和度", "排除灰白", C.blue, "#EEF4FC"],
    ["V", "明度", "排除過暗", C.green, "#EEF8EC"],
  ];
  hsvItems.forEach((item, idx) => {
    const x = 575 + idx * 210;
    addBox(slide, `hsv-box-${item[0]}`, x, 140, 192, 72, item[4], item[3]);
    addText(slide, `hsv-key-${item[0]}`, item[0], x + 10, 150, 42, 52, {
      fontSize: 26,
      bold: true,
      color: item[3],
      alignment: "center",
    });
    addText(slide, `hsv-copy-${item[0]}`, `${item[1]}\n${item[2]}`, x + 58, 150, 122, 52, {
      fontSize: 15,
      color: C.ink,
    });
  });
  const steps = [
    ["1", "限定圓形 ROI", "只分析培養皿內部"],
    ["2", "嚴格 HSV 種子", "H 15–49、S ≥ 54、V ≥ 30"],
    ["3", "黃色彩度延伸", "淡黃色必須與種子連通"],
    ["4", "清理並保留主體", "去除小雜訊與非主要連通區"],
  ];
  const nodes = steps.map((step, idx) => {
    const y = 238 + idx * 84;
    const node = addBox(slide, `algorithm-step-${idx + 1}`, 575, y, 612, 62, idx === 2 ? "#FFF3E8" : C.white, idx === 2 ? "#F3B37A" : C.line);
    slide.shapes.add({
      geometry: "ellipse",
      name: `algorithm-number-bg-${idx + 1}`,
      position: { left: 589, top: y + 12, width: 38, height: 38 },
      fill: C.navy,
      line: { style: "solid", fill: "none", width: 0 },
    });
    addText(slide, `algorithm-number-${idx + 1}`, step[0], 589, y + 12, 38, 38, {
      fontSize: 17,
      bold: true,
      color: C.white,
      alignment: "center",
    });
    addText(slide, `algorithm-title-${idx + 1}`, step[1], 646, y + 5, 260, 25, {
      fontSize: 18,
      bold: true,
      color: C.navy,
    });
    addText(slide, `algorithm-detail-${idx + 1}`, step[2], 646, y + 31, 520, 24, {
      fontSize: 14,
      color: C.muted,
    });
    return node;
  });
  for (let i = 0; i < nodes.length - 1; i += 1) {
    slide.shapes.connect(nodes[i + 1], nodes[i], {
      kind: "straight",
      fromSide: "top",
      toSide: "bottom",
      line: { style: "solid", fill: "#8EA0B8", width: 1.8 },
      head: { type: "triangle", width: "sm", length: "sm" },
    });
  }
  addBox(slide, "algorithm-principle", 575, 585, 612, 75, C.navy, C.navy);
  addText(slide, "algorithm-principle-text", "核心原則：先精準，再補漏\n影片驗證時，持續多幀出現的淡色分支才進入時間補回。", 596, 594, 570, 58, {
    fontSize: 17,
    bold: true,
    color: C.white,
    alignment: "center",
  });
  setNotes(slide,
    "HSV 的作用是把顏色拆成色相、飽和度與明度。色相先鎖定黃色，飽和度排除灰白背景，明度排除太暗的區域。程式先限制在圓形培養皿 ROI，再用 H 十五到四十九、S 大於等於五十四、V 大於等於三十建立高可信種子。淡黃色區域不會直接全部加入，而是用紅減藍、綠減藍的黃色彩度判斷，並要求它和種子連通。最後去除小物件並保留主要連通區。這個設計的重點是先精準再補漏，而不是直接放寬 HSV，因為直接放寬會把背景一起吃進來。時間補回只用在影片驗證流程，持續多幀出現的淡色分支才會加入。"
  );
}

// Slide 6
{
  const slide = presentation.slides.add();
  addHeader(slide, "HSV 後再加入時間共識，補回持續出現的淡色分支", "temporal recovery", 6);
  await addImage(slide, "temporal-comparison", "temporal-comparison-frame.png", { left: 56, top: 132, width: 1168, height: 292 }, {
    fit: "contain",
    alt: "Original frame, current HSV mask, and temporal recovery comparison; cyan marks added pixels",
  });
  addText(slide, "temporal-label-original", "原始影像", 56, 420, 360, 28, { fontSize: 17, bold: true, color: C.muted, alignment: "center" });
  addText(slide, "temporal-label-hsv", "單幀 HSV 遮罩", 460, 420, 360, 28, { fontSize: 17, bold: true, color: C.green, alignment: "center" });
  addText(slide, "temporal-label-final", "HSV + Temporal（青色＝補回）", 864, 420, 360, 28, { fontSize: 17, bold: true, color: C.blue, alignment: "center" });
  const temporalSteps = [
    ["1", "淡黃色候選", "Chroma ≥ 9，排除皿緣 30 px"],
    ["2", "5 幀時間共識", "至少 3 幀重複出現"],
    ["3", "與 HSV 主體連通", "距 strict mask 10 px 內"],
    ["4", "合併與平滑", "最大連通區 + 5 幀平滑"],
  ];
  const temporalNodes = temporalSteps.map((step, idx) => {
    const x = 56 + idx * 290;
    const node = addBox(slide, `temporal-step-${idx + 1}`, x, 470, 260, 92, idx === 2 ? "#FFF3E8" : C.white, idx === 2 ? "#F3B37A" : C.line);
    slide.shapes.add({ geometry: "ellipse", name: `temporal-number-bg-${idx + 1}`, position: { left: x + 14, top: 490, width: 38, height: 38 }, fill: C.navy, line: { style: "solid", fill: "none", width: 0 } });
    addText(slide, `temporal-number-${idx + 1}`, step[0], x + 14, 490, 38, 38, { fontSize: 17, bold: true, color: C.white, alignment: "center" });
    addText(slide, `temporal-title-${idx + 1}`, step[1], x + 62, 480, 184, 28, { fontSize: 18, bold: true, color: C.navy });
    addText(slide, `temporal-detail-${idx + 1}`, step[2], x + 62, 510, 184, 38, { fontSize: 13, color: C.muted });
    return node;
  });
  for (let i = 0; i < temporalNodes.length - 1; i += 1) {
    slide.shapes.connect(temporalNodes[i + 1], temporalNodes[i], { kind: "straight", fromSide: "left", toSide: "right", line: { style: "solid", fill: "#8EA0B8", width: 2 }, head: { type: "triangle", width: "sm", length: "sm" } });
  }
  addBox(slide, "temporal-takeaway", 56, 590, 1168, 70, C.navy, C.navy);
  addText(slide, "temporal-takeaway-text", "輸出遮罩 = 嚴格 HSV 主體 + 通過時間與連通條件的補回區域\n綠色為最終本體，青色顯示 temporal 新增像素，紅色為外邊界。", 80, 598, 1120, 54, { fontSize: 17, bold: true, color: C.white, alignment: "center" });
  setNotes(slide,
    "這一頁是建立在 HSV 之後的時間補回演算法。左邊是原始影像，中間是單幀 HSV 結果，右邊是加入 temporal recovery 之後的結果，青色代表新增的像素。程式先用較低門檻的黃色彩度找候選區，目前 chroma 最低為九、紅綠通道至少四十、最大亮度二百四十，並排除皿緣三十個像素及小於十五像素的雜訊。接著使用五幀視窗，像素至少在三幀出現才形成時間共識。候選連通區還必須接觸 HSV strict mask 向外擴張十個像素的範圍，避免把遠處背景加入。最後把 strict mask 與新增區域合併，只保留最大連通主體，再用五幀、至少三幀一致做最後平滑。這表示淡色分支不是因為單幀顏色相似就加入，而是同時通過時間一致性與空間連通性。"
  );
}

// Slide 7
{
  const slide = presentation.slides.add();
  addHeader(slide, "同一組規則可追蹤三個階段，下一步是定量驗證", "results and next steps", 7);
  const stageData = [
    ["image8.png", "Front", "小面積、高可信核心", "#EAF2FB"],
    ["image9.png", "Middle", "分支擴張、邊界變複雜", "#EEF8EC"],
    ["image10.png", "Rear", "接近皿緣、背景風險最高", "#FFF3E8"],
  ];
  for (const [idx, item] of stageData.entries()) {
    const x = 40 + idx * 410;
    await addImage(slide, `stage-image-${idx + 1}`, item[0], { left: x, top: 145, width: 380, height: 285 }, {
      crop: { left: 0.02, top: 0.02, right: 0.02, bottom: 0.02 },
      alt: `${item[1]} stage segmentation overlay`,
    });
    addBox(slide, `stage-caption-bg-${idx + 1}`, x, 442, 380, 84, item[3], C.line);
    addText(slide, `stage-title-${idx + 1}`, item[1], x + 18, 450, 110, 30, {
      fontSize: 22,
      bold: true,
      color: C.navy,
    });
    addText(slide, `stage-detail-${idx + 1}`, item[2], x + 18, 480, 340, 30, {
      fontSize: 16,
      color: C.ink,
    });
  }
  addBox(slide, "next-step-band", 40, 560, 1200, 112, C.navy, C.navy);
  addText(slide, "next-step-heading", "下一步", 64, 578, 120, 32, {
    fontSize: 22,
    bold: true,
    color: C.white,
  });
  addText(slide, "next-step-1", "① 凍結參數與程式版本", 210, 573, 280, 42, {
    fontSize: 17,
    bold: true,
    color: C.white,
  });
  addText(slide, "next-step-2", "② 以人工遮罩建立獨立測試集", 500, 573, 330, 42, {
    fontSize: 17,
    bold: true,
    color: C.white,
  });
  addText(slide, "next-step-3", "③ 報告 Dice、面積誤差與時間穩定性", 835, 573, 375, 42, {
    fontSize: 17,
    bold: true,
    color: C.white,
  });
  addText(slide, "next-step-note", "目前結果證明流程可運作；是否達到論文級準確度，必須由獨立測試資料回答。", 210, 620, 1000, 34, {
    fontSize: 15,
    color: "#DCE6F5",
    alignment: "center",
  });
  setNotes(slide,
    "最後用三個階段整理目前結果。Front 階段面積小、顏色集中，最容易得到高可信遮罩；Middle 階段開始出現大量細枝，邊界複雜度上升；Rear 階段接近培養皿邊緣，亮度、皿緣與背景最容易造成誤判。這些圖可以證明同一套流程能在不同階段運作，但還不能直接宣稱準確度已達標。下一步要先凍結程式與參數，再用未參與調參的人工遮罩做獨立測試，最後報告 Dice、面積誤差與時間穩定性。這樣才能把展示結果轉成老師與論文都能檢查的量化證據。"
  );
}

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${qaDir}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${qaDir}/${stem}.layout.json`, await layout.text(), "utf8");
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${qaDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));
const notesInspect = await presentation.inspect({ kind: "slide,notes,textbox,image,shape", maxChars: 30000 });
await fs.writeFile(`${qaDir}/inspect.ndjson`, notesInspect.ndjson, "utf8");
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPath);
console.log(JSON.stringify({ outputPath, slides: presentation.slides.items.length }, null, 2));






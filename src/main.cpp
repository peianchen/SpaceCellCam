/*******************************************************
 * XIAO ESP32S3 + Camera → SD（常駐版，無休眠，分桶）
 * - 固定週期：LED 預熱 → 丟棄首幀 → 拍照 → SD 寫入(atomic) → 驗證
 * - 檔名：/bXXX/image_<bootCount>.jpg（每 2000 張分一桶）
 * - 寫入：*.tmp → (若已有同名則刪除) → rename 到最終檔
 * - 分段寫入 + 讀回驗證 + 失敗降速重掛一次
 * - 週期不漂移（對齊到下一個邊界）
 *******************************************************/

#include "esp_camera.h"
#include "SPI.h"
#include "esp_timer.h"
#include <SdFat.h>

// ===== 相機腳位定義（Seeed XIAO ESP32S3）=====
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// ===== 參數 =====
#define TARGET_PERIOD_S  5            // 目標週期（秒）
#define LED_PIN          D1           // 補光 LED 腳位
#define SD_CS            21           // SD CS 腳位
#define BUCKET_SIZE      1000         // 每個資料夾最多檔案數

// ===== SdFat 物件 =====
SdFat32 sd;                           // FAT32 檔案系統（若用 64GB exFAT，建議改 SdFs）
File32  file;                         // 檔案物件
uint32_t g_sd_hz = 0;                 // 實際掛載的 SPI 頻率（Hz）

// ===== 連續運行計數（命名用）=====
int bootCount = 0;                    // 常駐運行，重新上電會從 0 開始

// ---------- 工具 ----------
static inline void setLEDBrightness(uint8_t b){ analogWrite(LED_PIN, b); }

// 以指定 MHz 嘗試初始化 SdFat（成功回傳 true）
bool sd_begin_mhz(uint32_t mhz) {
  SdSpiConfig cfg(SD_CS, SHARED_SPI, SD_SCK_MHZ(mhz));
  if (sd.begin(cfg)) {
    g_sd_hz = mhz * 1000000UL;
    Serial.printf("[SD] mounted at %u MHz\n", (unsigned)mhz);
    return true;
  }
  return false;
}

// 多個 SPI 時脈依序嘗試（先穩再快）
bool sd_init_fast() {
  const uint32_t cand_mhz[] = {25, 20, 16, 12};
  for (size_t i = 0; i < sizeof(cand_mhz)/sizeof(cand_mhz[0]); ++i) {
    if (sd_begin_mhz(cand_mhz[i])) return true;
  }
  return false;
}

// 丟棄第一幀（某些感光器第一幀不穩定）
static inline void discard_first_frame(){
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
  delay(20);
}

// 確保分桶資料夾存在（不存在就建立）
static bool sd_ensure_dir(const char* path) {
  if (sd.exists(path)) return true;
  if (sd.mkdir(path)) {
    Serial.printf("[SD] mkdir: %s\n", path);
    return true;
  }
  Serial.printf("[SD] mkdir failed: %s\n", path);
  return false;
}

/**
 * 以「*.tmp → 刪除同名舊檔 → rename」寫入 JPEG，並且讀回驗證
 */
bool atomicWriteAndVerify(const char* finalPath, const uint8_t* data, size_t len,
                          double* ms_out, double* mbps_out) {
  char tmpPath[64];
  snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", finalPath);

  // 1) 開暫存檔（覆蓋舊 tmp）
  File32 f = sd.open(tmpPath, O_CREAT | O_TRUNC | O_WRITE);
  if (!f) {
    Serial.println("[SD] open tmp failed");
    return false;
  }

  // 2) 預配置（失敗不致命）
  f.preAllocate(len);

  // 3) 計時 + 分段寫入
  int64_t t0 = esp_timer_get_time();
  const size_t CHUNK = 32 * 1024;
  size_t written = 0;
  while (written < len) {
    size_t n = (len - written) > CHUNK ? CHUNK : (len - written);
    size_t w = f.write(data + written, n);
    if (w != n) {
      Serial.printf("[SD] write short (%u/%u) at %u\n",
                    (unsigned)w, (unsigned)n, (unsigned)written);
      f.close();
      sd.remove(tmpPath);
      return false;
    }
    written += n;
  }

  // 4) 強制落盤
  if (!f.sync()) {
    Serial.println("[SD] sync failed");
    f.close();
    sd.remove(tmpPath);
    return false;
  }
  f.close();

  // 5) 若已有同名最終檔，先刪除（方案 A 的關鍵）
  if (sd.exists(finalPath)) {
    if (!sd.remove(finalPath)) {
      Serial.println("[SD] remove existing final failed");
      sd.remove(tmpPath);
      return false;
    }
  }

  // 6) 將 tmp → final（幾乎原子替換）
  if (!sd.rename(tmpPath, finalPath)) {
    Serial.println("[SD] rename tmp->final failed");
    sd.remove(tmpPath);
    return false;
  }

  // 7) 讀回驗證：確認檔案真的存在且大小正確
  File32 r = sd.open(finalPath, O_READ);
  if (!r) {
    Serial.println("[SD] reopen for verify failed");
    sd.remove(finalPath);
    return false;
  }
  uint64_t sz = r.size();
  r.close();
  if (sz != len) {
    Serial.printf("[SD] verify size mismatch: %llu (on card) vs %u (buf)\n",
                  (unsigned long long)sz, (unsigned)len);
    sd.remove(finalPath);
    return false;
  }

  // 輸出時間與吞吐量
  int64_t t1 = esp_timer_get_time();
  if (ms_out)   *ms_out   = (t1 - t0) / 1000.0;
  if (mbps_out) *mbps_out = (len / (1024.0 * 1024.0)) / ((*ms_out)/1000.0);
  return true;
}

// ---------- 拍照 + 存檔 ----------
void photo_save(const char *filePath) {
  int64_t t_led_on = esp_timer_get_time();

  // 1) LED 預熱（讓曝光穩定）
  setLEDBrightness(50);
  delay(200);

  // 2) 丟棄第一幀（避免未穩定畫面）
  discard_first_frame();

  // 3) 正式取幀
  camera_fb_t *fb = esp_camera_fb_get();
  int64_t t_got_fb = esp_timer_get_time();
  if (!fb) {
    Serial.println("[CAM] esp_camera_fb_get() failed");
    setLEDBrightness(0);
    return;
  }

  // ✅ 拍到幀後立即關燈（存檔不需補光）
  setLEDBrightness(0);
  int64_t t_led_off = esp_timer_get_time();

  // 4) 寫入（原子替換）+ 讀回驗證；失敗則降速重掛再試一次
  double write_ms = 0.0, write_mbps = 0.0;
  bool ok = atomicWriteAndVerify(filePath, fb->buf, fb->len, &write_ms, &write_mbps);
  if (!ok) {
    Serial.println("[SD] write+verify failed, remount slower and retry once...");
    sd.end();
    // 依序試 20 → 16 → 12 MHz
    if (sd_begin_mhz(20) || sd_begin_mhz(16) || sd_begin_mhz(12)) {
      ok = atomicWriteAndVerify(filePath, fb->buf, fb->len, &write_ms, &write_mbps);
    }
  }

  // 5) 釋放影像緩衝
  esp_camera_fb_return(fb);
  int64_t t_after_fb_return = esp_timer_get_time();

  // 6) 計時資訊（debug）
  double led_to_fb_ms    = (t_got_fb - t_led_on) / 1000.0;
  double fb_to_ledoff_ms = (t_led_off - t_got_fb) / 1000.0; // 幾乎為 0
  double fb_to_done_ms   = (t_after_fb_return - t_got_fb) / 1000.0;

  Serial.printf("Timings  LED->FB: %.1f ms, FB->LEDoff: %.1f ms, FB->SD(write): %.1f ms\n",
                led_to_fb_ms, fb_to_ledoff_ms, write_ms);
  Serial.printf("Size: %.2f MB,   SD throughput: %.2f MB/s  (SPI %.0f MHz)\n",
                fb->len / (1024.0 * 1024.0), write_mbps, g_sd_hz / 1e6);

  if (ok) Serial.printf("[OK] Saved picture: %s\n", filePath);
  else    Serial.printf("[FAIL] Save failed: %s\n", filePath);

  // 7) 一點喘息時間
  delay(10);
}

void setup() {
  Serial.begin(115200);
  delay(100);

  // 啟動 SPI
  SPI.begin();

  pinMode(LED_PIN, OUTPUT);
  setLEDBrightness(0);

  // ===== 相機設定（穩定優先）=====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;

  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 5000000;            // 目前設 5 MHz；若要更快可改 10000000
  config.frame_size   = FRAMESIZE_QSXGA;     // 有 PSRAM 時可用；若時間吃緊先降到 UXGA/SXGA
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count     = 1;

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.fb_count  = 2;                  // 雙緩衝
      config.grab_mode = CAMERA_GRAB_LATEST; // 取最新幀
    } else {
      config.frame_size  = FRAMESIZE_SVGA;   // 無 PSRAM 降級
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[CAM] init failed");
    while (true) { delay(1000); }
  }

  // 可依場景把 AE/AWB 鎖住以提升一致性
  sensor_t *s = esp_camera_sensor_get();
  s->set_whitebal(s, 1);
  s->set_wb_mode(s, 1);

  // 固定曝光/增益（避免暗場拉長曝光）
  s->set_exposure_ctrl(s, 0);
  s->set_gain_ctrl(s, 0);
  s->set_aec_value(s, 600);
  s->set_agc_gain(s, 8);
  s->set_aec2(s, 0);

  // ===== 掛載 SD（多時脈嘗試）=====
  if (!sd_init_fast()) {
    Serial.println("[SD] mount failed (all speeds tried).");
    while (true) { delay(1000); }
  }

  Serial.println("Continuous mode: capture (no sleep).");
}

void loop() {
  static uint32_t period_ms = TARGET_PERIOD_S * 1000UL;
  static uint32_t next_tick = millis();  // 對齊時脈，避免漂移

  ++bootCount;

  // ---- 分桶：每 2000 張一個資料夾 /bXXX ----
  int bucket = (bootCount - 1) / BUCKET_SIZE;      // 0,1,2,...
  char folder[32];
  snprintf(folder, sizeof(folder), "/b%03d", bucket);

  // 僅在換桶時建立資料夾，避免每次 exists 開銷
  static int last_bucket = -1;
  if (bucket != last_bucket) {
    if (!sd_ensure_dir(folder)) {
      Serial.println("[SD] cannot ensure bucket folder, skip this shot.");
      // 仍可拍但不存檔：直接進下一輪
    }
    last_bucket = bucket;
  }

  // 最終檔名：/bXXX/image_<bootCount>.jpg
  char filename[64];
  snprintf(filename, sizeof(filename), "%s/image_%d.jpg", folder, bootCount);

  // 若想每桶從 1 開始：改用以下兩行即可
  // int idx_in_bucket = (bootCount - 1) % BUCKET_SIZE + 1;
  // snprintf(filename, sizeof(filename), "%s/image_%d.jpg", folder, idx_in_bucket);

  photo_save(filename);

  // === 精準週期控制：對齊下一個邊界 ===
  next_tick += period_ms;
  int32_t remain = (int32_t)(next_tick - millis());
  if (remain > 0) {
    delay(remain);  // 阻塞到下一次時間點
  } else {
    // 若拍照/存檔超時，立刻進下一輪，但維持相位（不重置 next_tick）
    // （也可改成：next_tick = millis(); 從現在起算）
  }
}

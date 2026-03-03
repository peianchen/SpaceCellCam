/*******************************************************
 * XIAO ESP32S3 + Camera → SD（常駐版，無休眠，分桶｜LED 用 analogWrite）
 * - 固定週期：LED 預熱 → 丟棄兩幀 → 拍照 → SD 寫入(atomic) → 驗證
 * - 分桶：/bXXX/image_<bootCount>.jpg（每 BUCKET_SIZE 張分一桶）
 * - 關鍵：單緩衝 + WHEN_EMPTY（避免 LED 未穩定時就開始曝光造成上半部偏暗）
 *******************************************************/

#include "esp_camera.h"
#include "SPI.h"
#include "esp_timer.h"
#include <SdFat.h>

// ===== 相機腳位（Seeed XIAO ESP32S3）=====
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// ===== 參數 =====
#define TARGET_PERIOD_S   10         // 週期（秒）
#define LED_PIN           D1        // 補光 LED 腳位
#define SD_CS             21        // SD CS 腳位
#define BUCKET_SIZE       500      // 每個資料夾最多檔案數
#define XCLK_FREQ_HZ      5000000   // 5 MHz（穩；可視情況試 20 MHz）
#define LED_PREHEAT_MS    200       // LED 預熱
#define LED_BRIGHTNESS    5        // LED 亮度（0~255）

// ===== SdFat 物件 =====
SdFat32 sd;
File32  file;
uint32_t g_sd_hz = 0;

// ===== 計數 =====
int bootCount = 0;

// ---------- LED ----------
static inline void setLEDBrightness(uint8_t b){ analogWrite(LED_PIN, b); }

// ---------- SD 掛載 ----------
bool sd_begin_mhz(uint32_t mhz) {
  SdSpiConfig cfg(SD_CS, SHARED_SPI, SD_SCK_MHZ(mhz));
  if (sd.begin(cfg)) { g_sd_hz = mhz * 1000000UL; Serial.printf("[SD] mounted at %u MHz\n", (unsigned)mhz); return true; }
  return false;
}
bool sd_init_fast() {
  const uint32_t cand_mhz[] = {25, 20, 16, 12};
  for (size_t i=0;i<sizeof(cand_mhz)/sizeof(cand_mhz[0]);++i) if (sd_begin_mhz(cand_mhz[i])) return true;
  return false;
}

// ---------- 丟棄一幀 ----------
static inline void discard_one_frame(){
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
}

// ---------- 分桶資料夾 ----------
static bool sd_ensure_dir(const char* path) {
  if (sd.exists(path)) return true;
  if (sd.mkdir(path)) { Serial.printf("[SD] mkdir: %s\n", path); return true; }
  Serial.printf("[SD] mkdir failed: %s\n", path); return false;
}

// ---------- Atomic 寫入 + 驗證 ----------
bool atomicWriteAndVerify(const char* finalPath, const uint8_t* data, size_t len,
                          double* ms_out, double* mbps_out) {
  char tmpPath[64];
  snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", finalPath);

  File32 f = sd.open(tmpPath, O_CREAT | O_TRUNC | O_WRITE);
  if (!f) { Serial.println("[SD] open tmp failed"); return false; }

  f.preAllocate(len); // 失敗不致命

  int64_t t0 = esp_timer_get_time();
  const size_t CHUNK = 32 * 1024;
  size_t written = 0;
  while (written < len) {
    size_t n = (len - written) > CHUNK ? CHUNK : (len - written);
    size_t w = f.write(data + written, n);
    if (w != n) { Serial.printf("[SD] write short (%u/%u) at %u\n",(unsigned)w,(unsigned)n,(unsigned)written);
      f.close(); sd.remove(tmpPath); return false; }
    written += n;
  }

  if (!f.sync()) { Serial.println("[SD] sync failed"); f.close(); sd.remove(tmpPath); return false; }
  f.close();

  if (sd.exists(finalPath) && !sd.remove(finalPath)) { Serial.println("[SD] remove final failed"); sd.remove(tmpPath); return false; }
  if (!sd.rename(tmpPath, finalPath)) { Serial.println("[SD] rename tmp->final failed"); sd.remove(tmpPath); return false; }

  File32 r = sd.open(finalPath, O_READ);
  if (!r) { Serial.println("[SD] reopen final failed"); sd.remove(finalPath); return false; }
  uint64_t sz = r.size(); r.close();
  if (sz != len) { Serial.printf("[SD] verify size mismatch: %llu vs %u\n",(unsigned long long)sz,(unsigned)len); sd.remove(finalPath); return false; }

  int64_t t1 = esp_timer_get_time();
  if (ms_out)   *ms_out = (t1 - t0) / 1000.0;
  if (mbps_out) *mbps_out = (len / (1024.0*1024.0)) / ((*ms_out)/1000.0);
  return true;
}

// ---------- 拍照 + 存檔（LED 先穩定，丟 2 幀再拍） ----------
void photo_save(const char *filePath) {
  int64_t t_led_on = esp_timer_get_time();

  // 1) LED 預熱與穩定
  setLEDBrightness(LED_BRIGHTNESS);
  delay(LED_PREHEAT_MS);

  // 2) 丟 2 幀（確保下一幀的曝光起點在 LED 穩定後）
  discard_one_frame();
  discard_one_frame();

  // 3) 正式取幀
  camera_fb_t *fb = esp_camera_fb_get();
  int64_t t_got_fb = esp_timer_get_time();
  if (!fb) {
    Serial.println("[CAM] esp_camera_fb_get() failed");
    setLEDBrightness(0);
    return;
  }

  // 4) 關燈
  setLEDBrightness(0);
  int64_t t_led_off = esp_timer_get_time();

  // 5) 寫入（原子）+ 驗證；失敗降速重掛一次
  double write_ms = 0.0, write_mbps = 0.0;
  bool ok = atomicWriteAndVerify(filePath, fb->buf, fb->len, &write_ms, &write_mbps);
  if (!ok) {
    Serial.println("[SD] write+verify failed, remount slower and retry once...");
    sd.end();
    if (sd_begin_mhz(20) || sd_begin_mhz(16) || sd_begin_mhz(12)) {
      ok = atomicWriteAndVerify(filePath, fb->buf, fb->len, &write_ms, &write_mbps);
    }
  }

  // 6) 釋放影像緩衝
  esp_camera_fb_return(fb);
  int64_t t_after_fb_return = esp_timer_get_time();

  // 7) 時間資訊
  double led_to_fb_ms    = (t_got_fb - t_led_on) / 1000.0;
  double fb_to_ledoff_ms = (t_led_off - t_got_fb) / 1000.0;
  Serial.printf("Timings  LED->FB: %.1f ms, FB->LEDoff: %.1f ms, FB->SD(write): %.1f ms\n",
                led_to_fb_ms, fb_to_ledoff_ms, write_ms);
  Serial.printf("Size: %.2f MB, SD: %.2f MB/s  (SPI %.0f MHz)\n",
                fb->len / (1024.0 * 1024.0), write_mbps, g_sd_hz / 1e6);

  if (ok) Serial.printf("[OK] Saved: %s\n", filePath);
  else    Serial.printf("[FAIL] Save failed: %s\n", filePath);

  delay(10);
}

void setup() {
  Serial.begin(115200);
  delay(100);

  SPI.begin();
  pinMode(LED_PIN, OUTPUT);
  setLEDBrightness(0);

  // ===== 相機設定（單緩衝 + WHEN_EMPTY 是關鍵）=====
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

  config.xclk_freq_hz = XCLK_FREQ_HZ;
  config.frame_size   = FRAMESIZE_QSXGA;    // 需要更穩就先降 UXGA/SXGA FRAMESIZE_QSXGA
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY; // 等待下一幀（避免舊幀）
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count     = 1;                  // ⭐ 單緩衝（避免 LATEST 帶來的時序不確定）

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[CAM] init failed");
    while (true) { delay(1000); }
  }

  // 固定白平衡/曝光/增益，避免暗場拉長曝光
  sensor_t *s = esp_camera_sensor_get();
  //s->set_brightness(s, -1);     // -2 to 2
  //s->set_contrast(s, 1);       // -2 to 2
  //s->set_whitebal(s, 1);
  s->set_wb_mode(s, 1);
  //s->set_exposure_ctrl(s, 0);
  //s->set_aec2(s, 0);
  //s->set_aec_value(s, 400);   // 依 LED 亮度調 400~800
  //s->set_gain_ctrl(s, 0);
  //s->set_agc_gain(s, 8);      // 5~12 常見
  s->set_brightness(s, 0);     // -2 to 2
  s->set_contrast(s, 2);       // -2 to 2
  s->set_saturation(s, 1);     // -2 to 2
  s->set_special_effect(s, 0); // 0 to 6 (0 - No Effect, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia)
  s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
  s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
  //s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
  s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
  s->set_aec2(s, 0);           // 0 = disable , 1 = enable
  s->set_ae_level(s, 0);       // -2 to 2
  s->set_aec_value(s, 300);    // 0 to 1200
  s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
  s->set_agc_gain(s, 0);       // 0 to 30
  s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
  s->set_bpc(s, 0);            // 0 = disable , 1 = enable
  s->set_wpc(s, 1);            // 0 = disable , 1 = enable
  s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
  s->set_lenc(s, 1);           // 0 = disable , 1 = enable
  s->set_dcw(s, 1);            // 0 = disable , 1 = enable
  s->set_colorbar(s, 0);       // 0 = disable , 1 = enable

  // 掛載 SD（多時脈嘗試）
  if (!sd_init_fast()) {
    Serial.println("[SD] mount failed (all speeds tried).");
    while (true) { delay(1000); }
  }

  Serial.println("Continuous mode: capture (no sleep).");
}

void loop() {
  static const uint32_t period_ms = TARGET_PERIOD_S * 1000UL;
  static uint32_t next_tick = millis();

  ++bootCount;

  // ---- 分桶：/bXXX ----
  int bucket = (bootCount - 1) / BUCKET_SIZE;
  char folder[32]; snprintf(folder, sizeof(folder), "/b%03d", bucket);
  static int last_bucket = -1;
  if (bucket != last_bucket) {
    sd_ensure_dir(folder);
    last_bucket = bucket;
  }

  // 檔名
  char filename[64];
  snprintf(filename, sizeof(filename), "%s/image_%d.jpg", folder, bootCount);
  // 每桶從 1 開始可改：
  // int idx = (bootCount - 1) % BUCKET_SIZE + 1;
  // snprintf(filename, sizeof(filename), "%s/image_%d.jpg", folder, idx);

  photo_save(filename);

  // 對齊下一個邊界
  next_tick += period_ms;
  int32_t remain = (int32_t)(next_tick - millis());
  if (remain > 0) delay(remain);

  // 追相位：若本輪超時，跳槽位追回
  int32_t behind = (int32_t)(millis() - next_tick);
  if (behind > 0) {
    uint32_t skip = (behind / (int32_t)period_ms) + 1;
    next_tick += skip * period_ms;
    Serial.printf("[SCHED] overrun %ld ms, skip %u slot(s) to realign.\n",
                  (long)behind, (unsigned)skip);
  }
}

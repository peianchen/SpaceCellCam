/*******************************************************
 * XIAO ESP32S3 + Camera → SD（SdFat 方案A：先刪再換）
 * - 10 秒週期：喚醒 → LED預熱 → 丟棄第一幀 → 拍照 → SD寫入
 * - 檔名：/image_<bootCount>.jpg（Deep Sleep 每次 +1）
 * - 寫入：*.tmp → (若已有同名則刪除) → rename 到最終檔
 * - 分段寫入 + 讀回驗證 + 失敗降速重掛一次 + 關檔後微延遲
 *******************************************************/

#include "esp_camera.h"
#include "SPI.h"
#include "esp_timer.h"
#include <SdFat.h>

// ===== 相機腳位定義（Seeed XIAO ESP32S3）=====
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// ===== 參數 =====
#define uS_TO_S_FACTOR   1000000ULL
#define TARGET_PERIOD_S  10           // 目標週期：10 秒
#define LED_PIN          D1           // 補光 LED 腳位
#define SD_CS            21           // SD CS 腳位

// ===== SdFat 物件 =====
SdFat32 sd;                           // FAT32 檔案系統
File32  file;                         // 檔案物件
uint32_t g_sd_hz = 0;                 // 實際掛載的 SPI 頻率（Hz）

// ===== Deep Sleep 計數（命名用）=====
RTC_DATA_ATTR int bootCount = 0;

// ---------- 工具 ----------
static inline void setLEDBrightness(uint8_t b){ analogWrite(LED_PIN, b); }

// 印出喚醒原因（debug 用）
void print_wakeup_reason() {
  esp_sleep_wakeup_cause_t r = esp_sleep_get_wakeup_cause();
  if (r == ESP_SLEEP_WAKEUP_TIMER) Serial.println("Wakeup caused by timer");
  else Serial.printf("Wakeup was not caused by deep sleep: %d\n", r);
}

// 以指定 MHz 嘗試初始化 SdFat（成功回傳 true）
bool sd_begin_mhz(uint32_t mhz) {
  // SHARED_SPI：與其他裝置共享 SPI；SD_SCK_MHZ 設定 SCK 頻率
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

/**
 * 以「*.tmp → 刪除同名舊檔 → rename」寫入 JPEG，並且讀回驗證
 * - finalPath：最終檔案路徑（例如 "/image_7.jpg"）
 * - data/len：JPEG 緩衝區
 * - ms_out/mbps_out：輸出寫入耗時與吞吐量（可為 nullptr）
 *
 * 流程：
 *  1. 打開 tmp 檔（O_CREAT|O_TRUNC|O_WRITE）
 *  2. 預配置容量（非必須，但可減少碎片）
 *  3. 分段寫入（32KB chunk）
 *  4. sync + close，保證落盤
 *  5. 若 final 已存在 → 先 remove(final)
 *  6. rename(tmp → final) 使替換幾乎原子
 *  7. 重新打開 final 檔讀回，驗證 size 一致
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
void photo_save(const char *fileName) {
  int64_t t_led_on = esp_timer_get_time();

  // 1) LED 預熱（讓曝光穩定）
  setLEDBrightness(50);
  delay(120);

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

  // 4) 寫入（原子替換）+ 讀回驗證；失敗則降速重掛再試一次
  double write_ms = 0.0, write_mbps = 0.0;
  bool ok = atomicWriteAndVerify(fileName, fb->buf, fb->len, &write_ms, &write_mbps);
  if (!ok) {
    Serial.println("[SD] write+verify failed, remount slower and retry once...");
    sd.end();
    // 依序試 20 → 16 → 12 MHz
    if (sd_begin_mhz(20) || sd_begin_mhz(16) || sd_begin_mhz(12)) {
      ok = atomicWriteAndVerify(fileName, fb->buf, fb->len, &write_ms, &write_mbps);
    }
  }

  // 5) 釋放影像緩衝
  esp_camera_fb_return(fb);
  int64_t t_led_off = esp_timer_get_time();

  // 6) 計時資訊（debug）
  double led_to_fb_ms = (t_got_fb - t_led_on) / 1000.0;
  double sd_to_off_ms = (t_led_off - (t_got_fb + (int64_t)(write_ms*1000))) / 1000.0;

  Serial.printf("Timings  LED->FB: %.1f ms, FB->SD(write): %.1f ms, SD->LEDoff: ~%.1f ms\n",
                led_to_fb_ms, write_ms, sd_to_off_ms);
  Serial.printf("Size: %.2f MB,   SD throughput: %.2f MB/s  (SPI %.0f MHz)\n",
                fb->len / (1024.0 * 1024.0), write_mbps, g_sd_hz / 1e6);

  if (ok) Serial.printf("[OK] Saved picture: %s\n", fileName);
  else    Serial.printf("[FAIL] Save failed: %s\n", fileName);

  // 7) 一點喘息時間（某些卡韌體保守，這樣更穩）
  delay(10);
}

void setup() {
  Serial.begin(115200);
  delay(100);

  // 啟動 SPI（若與其他裝置共用 SPI，仍建議呼叫）
  SPI.begin();

  pinMode(LED_PIN, OUTPUT);
  setLEDBrightness(0);

  int64_t t_awake_start = esp_timer_get_time();

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

  config.xclk_freq_hz = 5000000;            // 10 MHz：穩定
  config.frame_size   = FRAMESIZE_QSXGA;     // 有 PSRAM 時可用
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
    esp_sleep_enable_timer_wakeup(TARGET_PERIOD_S * uS_TO_S_FACTOR);
    esp_deep_sleep_start();
  }

  // 可依場景把 AE/AWB 鎖住以提升一致性
  sensor_t *s = esp_camera_sensor_get();
  s->set_wb_mode(s, 1);
  //s->set_gain_ctrl(s, 0);      // 關自動增益
  //s->set_exposure_ctrl(s, 0);  // 關自動曝光
  //s->set_aec_value(s, 600);    // 固定曝光值，可調 400~800 視亮度
  //s->set_agc_gain(s, 8);       // 固定增益，可調 5~12
  //s->set_brightness(s, 1);     // 提亮
  //s->set_contrast(s, 1);       // 增加對比
  //s->set_whitebal(s, 0);       // 關自動白平衡



  // ===== 掛載 SD（多時脈嘗試）=====
  if (!sd_init_fast()) {
    Serial.println("[SD] mount failed (all speeds tried).");
    esp_sleep_enable_timer_wakeup(TARGET_PERIOD_S * uS_TO_S_FACTOR);
    esp_deep_sleep_start();
  }

  // ===== 檔名：使用 bootCount 確保序號（同名會覆蓋）=====
  ++bootCount;  // 每次喚醒 +1
  Serial.printf("Boot number: %d\n", bootCount);
  print_wakeup_reason();

  char filename[48];
  sprintf(filename, "/image_%d.jpg", bootCount);

  // ===== 拍照 + 存檔 =====
  photo_save(filename);

  // ===== 週期補償，進入 Deep Sleep =====
  int64_t awake_us  = esp_timer_get_time() - t_awake_start;
  int64_t target_us = (int64_t)TARGET_PERIOD_S * uS_TO_S_FACTOR;
  int64_t sleep_us  = target_us - awake_us;
  if (sleep_us < 1000) sleep_us = 1000;

  Serial.printf("Awake time: %.1f ms, Sleep for: %.1f ms (target = %d s)\n",
                awake_us / 1000.0, sleep_us / 1000.0, TARGET_PERIOD_S);

  esp_sleep_enable_timer_wakeup(sleep_us);
  Serial.println("Going to sleep now");
  Serial.flush();
  esp_deep_sleep_start();
}

void loop() { /* 不會到這裡 */ }

#include "esp_camera.h"
#include "SPI.h"
#include "esp_timer.h"

// ===== SdFat =====
#include <SdFat.h>

// ===== 相機腳位定義（XIAO ESP32S3）=====
#define CAMERA_MODEL_XIAO_ESP32S3  // Has PSRAM
#include "camera_pins.h"

// ===== 參數 =====
#define uS_TO_S_FACTOR 1000000ULL
#define TARGET_PERIOD_S 10                  // 目標週期：10 秒
#define LED_PIN D1                          // 補光 LED 腳位
#define SD_CS   21                          // 你的 SD CS 腳位

// ===== SdFat 物件（v2：SdFat32 + File32）=====
SdFat32 sd;         // FAT32 檔案系統
File32  file;       // 檔案物件
uint32_t g_sd_hz = 0;

RTC_DATA_ATTR int bootCount = 0;

// ===== 工具 =====
void setLEDBrightness(uint8_t b) { analogWrite(LED_PIN, b); }

void print_wakeup_reason() {
  esp_sleep_wakeup_cause_t r = esp_sleep_get_wakeup_cause();
  if (r == ESP_SLEEP_WAKEUP_TIMER) Serial.println("Wakeup caused by timer");
  else Serial.printf("Wakeup was not caused by deep sleep: %d\n", r);
}

// 以指定 MHz 嘗試初始化 SdFat（回傳是否成功）
bool sd_begin_mhz(uint32_t mhz) {
  // SHARED_SPI：與其他裝置共享 SPI 匯流排；SD_SCK_MHZ(mhz) 設定 SCK
  SdSpiConfig cfg(SD_CS, SHARED_SPI, SD_SCK_MHZ(mhz));
  if (sd.begin(cfg)) {
    g_sd_hz = mhz * 1000000UL;
    Serial.printf("SdFat mounted at %.2f MHz\n", (double)mhz);
    return true;
  }
  return false;
}

// 依序嘗試多個 SPI 時脈，選第一個成功者
bool sd_init_fast() {
  const uint32_t cand_mhz[] = {50, 40, 25, 20, 16, 12}; // ESP32 常見上限 50 MHz
  for (size_t i = 0; i < sizeof(cand_mhz)/sizeof(cand_mhz[0]); ++i) {
    if (sd_begin_mhz(cand_mhz[i])) return true;
  }
  return false;
}

// ===== 寫檔（預配置 + 一次性寫入）=====
bool writeJpegFast(const char* path, const uint8_t* data, size_t len, double* ms_out, double* mbps_out) {
  // 以 O_CREAT | O_TRUNC | O_WRITE 開啟；v2 回傳 File32
  file = sd.open(path, O_CREAT | O_TRUNC | O_WRITE);
  if (!file) {
    Serial.println("Failed to open file");
    return false;
  }

  // 預配置容量，降低 FAT 更新/碎片化（若失敗不致命）
  if (!file.preAllocate(len)) {
    Serial.println("preAllocate() failed (continuing without it)");
  }

  int64_t t0 = esp_timer_get_time();
  size_t w = file.write(data, len);
  file.sync();   // 落盤
  int64_t t1 = esp_timer_get_time();

  file.close();

  if (w != len) {
    Serial.printf("Write failed (wrote %u/%u)\n", (unsigned)w, (unsigned)len);
    return false;
  }

  double ms = (t1 - t0) / 1000.0;
  double mb = len / (1024.0 * 1024.0);
  double mbps = (ms > 0.0) ? (mb / (ms / 1000.0)) : 0.0;

  if (ms_out)   *ms_out = ms;
  if (mbps_out) *mbps_out = mbps;

  return true;
}

// ===== 拍照流程：LED 預熱 → 丟棄一幀 → 正式取幀 → SdFat 快速寫入 → 統計 =====
void photo_save(const char *fileName) {
  int64_t t_led_on = esp_timer_get_time();

  // 1) LED 預熱（100–150 ms）
  setLEDBrightness(60);
  delay(120);

  // 3) 正式取幀
camera_fb_t *fb = esp_camera_fb_get();
int64_t t_got_fb = esp_timer_get_time();
if (!fb) {
  Serial.println("Failed to get camera frame buffer");
  setLEDBrightness(0);
  return;
}

// ✅ 拍到畫面後立即關燈，存檔不需要補光
setLEDBrightness(0);

// 4) 寫入（SdFat）
double write_ms = 0.0, write_mbps = 0.0;
bool ok = writeJpegFast(fileName, fb->buf, fb->len, &write_ms, &write_mbps);

// 5) 收尾
esp_camera_fb_return(fb);
int64_t t_led_off = esp_timer_get_time();


  // 6) 統計
  double led_to_fb_ms = (t_got_fb - t_led_on) / 1000.0;
  double sd_to_off_ms = (t_led_off - (t_got_fb + (int64_t)(write_ms*1000))) / 1000.0;

  Serial.printf("Timings  LED->FB: %.1f ms, FB->SD: %.1f ms, SD->LEDoff: ~%.1f ms\n",
                led_to_fb_ms, write_ms, sd_to_off_ms);
  Serial.printf("Size: %.2f MB,   SD throughput: %.2f MB/s  (SPI %.2f MHz)\n",
                fb->len / (1024.0 * 1024.0), write_mbps, g_sd_hz / 1e6);

  if (ok) Serial.printf("Saved picture: %s\n", fileName);
  else    Serial.printf("Save failed: %s\n", fileName);
}

void setup() {
  Serial.begin(115200);
  delay(100);

  // 啟動 SPI（若用預設腳位，這行即可）
  SPI.begin();

  pinMode(LED_PIN, OUTPUT);
  setLEDBrightness(0);

  int64_t t_awake_start = esp_timer_get_time();

  // ===== 相機設定（維持畫質與 XCLK）=====
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

  config.xclk_freq_hz = 10000000;            // 保持 10 MHz
  config.frame_size   = FRAMESIZE_QSXGA;     // 保持 QSXGA
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;                  // 保持 10
  config.fb_count     = 1;

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.fb_count  = 2;                  // 雙幀緩衝
      config.grab_mode = CAMERA_GRAB_LATEST; // 取最新幀
    } else {
      config.frame_size  = FRAMESIZE_SVGA;   // 無 PSRAM 才降級
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    int64_t sleep_us = (int64_t)TARGET_PERIOD_S*uS_TO_S_FACTOR - (esp_timer_get_time()-t_awake_start);
    if (sleep_us < 1000) sleep_us = 1000;
    esp_sleep_enable_timer_wakeup(sleep_us);
    esp_deep_sleep_start();
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_wb_mode(s, 1);  // 依你原設定；若光源固定，未來可考慮固定 AE/AWB

  // ===== SdFat 初始化（多時脈嘗試）=====
  if (!sd_init_fast()) {
    Serial.println("SdFat mount failed (all speeds tried).");
    int64_t sleep_us = (int64_t)TARGET_PERIOD_S*uS_TO_S_FACTOR - (esp_timer_get_time()-t_awake_start);
    if (sleep_us < 1000) sleep_us = 1000;
    esp_sleep_enable_timer_wakeup(sleep_us);
    esp_deep_sleep_start();
    return;
  }

  // 卡片資訊（新版 cid_t 沒有 prv_n/prv_m，要從 prv 拆 major/minor）
  if (sd.card()) {
    cid_t cid;
    if (sd.card()->readCID(&cid)) {
      uint8_t prv_major = (cid.prv >> 4) & 0x0F;
      uint8_t prv_minor = cid.prv & 0x0F;
      Serial.printf("SD MID: 0x%02X  OID: %c%c  PRV: %u.%u\n",
                    cid.mid, cid.oid[0], cid.oid[1], prv_major, prv_minor);
    }
  }

  ++bootCount;
  Serial.println("Boot number: " + String(bootCount));
  print_wakeup_reason();

  // 檔名：用 μs 時戳避免覆寫
  char filename[48];
  sprintf(filename, "/image_%d.jpg", bootCount);

  photo_save(filename);

  // ===== 10 秒週期補償 =====
  int64_t awake_us  = esp_timer_get_time() - t_awake_start;
  int64_t target_us = (int64_t)TARGET_PERIOD_S * uS_TO_S_FACTOR;
  int64_t sleep_us  = target_us - awake_us;
  if (sleep_us < 1000) sleep_us = 1000;

  Serial.printf("Awake time: %.1f ms, Sleep for: %.1f ms (target period = %d s)\n",
                awake_us / 1000.0, sleep_us / 1000.0, TARGET_PERIOD_S);

  esp_sleep_enable_timer_wakeup(sleep_us);
  Serial.println("Going to sleep now");
  Serial.flush();
  esp_deep_sleep_start();
}

void loop() { /* 不會到這裡 */ }

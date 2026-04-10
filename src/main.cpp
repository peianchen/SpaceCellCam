/*******************************************************
 * XIAO ESP32S3 + Camera → SD（常駐版，無休眠，分桶｜LED 用 LEDC 穩定 PWM）
 * - 固定週期：LED 預熱 → 丟棄兩幀 → 拍照 → SD 寫入(atomic) → 驗證
 * - 分桶：/bXXX/image_<count>.jpg（每 BUCKET_SIZE 張分一桶）
 * - 關鍵：單緩衝 + WHEN_EMPTY（避免 LED 未穩定時就開始曝光造成上半部偏暗）
 *
 * 修正：
 * (1) 避免 analogWrite 與 camera LEDC 衝突：改用 LEDC(獨立 channel/timer)
 * (2) 修正排程 overrun 判定：只在落後超過 1 個週期才 skip（加容忍）
 * (3) 修正 use-after-free：fb return 前先存 fb->len 等資訊
 *
 * 曝光模式：
 * EXPOSURE_MODE = 0: 自動曝光/自動增益
 * EXPOSURE_MODE = 1: 固定曝光/固定增益（較一致）
 *******************************************************/

#include "esp_camera.h"
#include "SPI.h"
#include "esp_timer.h"
#include <SdFat.h>
#include "ov5640_af_support.h"

// ===== 相機腳位（Seeed XIAO ESP32S3）=====
#ifndef CAMERA_MODEL_XIAO_ESP32S3
#define CAMERA_MODEL_XIAO_ESP32S3
#endif
#include "camera_pins.h"

// ===== 參數 =====
#define TARGET_PERIOD_S   10         // 週期（秒）
#define LED_PIN           D1         // 補光 LED 腳位
#define SD_CS             21         // SD CS 腳位
#define BUCKET_SIZE       500        // 每個資料夾最多檔案數
#define XCLK_FREQ_HZ      5000000    // 5 MHz（穩；可視情況試 20 MHz）
#define LED_PREHEAT_MS    200        // LED 預熱
#define LED_BRIGHTNESS    10         // LED 亮度（0~255）
#define ENABLE_AUTOFOCUS  1          // 只在 OV5640 + AF 模組時生效
#define AF_SETUP_SETTLE_MS 1500      // setup 時給 AF 收斂的時間
#define AF_POLL_MS        50         // setup 時輪詢 AF 狀態間隔
#define AF_STABLE_COUNT   3          // 連續幾次穩定才視為已對焦

// ===== 曝光/增益模式切換 =====
#define EXPOSURE_MODE     0          // 0=自動  1=固定（建議量測一致性）

// ===== 固定曝光/增益參數（EXPOSURE_MODE=1 時生效）=====
#define FIX_AEC_VALUE     100        // 0~1200（曝光）
#define FIX_AGC_GAIN      1          // 0~30（增益，常用 5~12）
#define FIX_GAIN_CEIL     2          // 0~6（增益天花板，2~4 常見）

// ===== LEDC for LED (避免與 camera LEDC 衝突) =====
// camera_config 用 CH0/T0；這裡用 CH4/T1
static const int LEDC_LED_CH    = 4;
static const int LEDC_LED_TIMER = 1; // (這個常數僅註解用，Arduino core 不直接用)
static const int LEDC_LED_FREQ  = 5000;
static const int LEDC_LED_RES   = 8;

// ===== SdFat 物件 =====
SdFat32 sd;
File32  file;
uint32_t g_sd_hz = 0;
bool g_af_supported = false;
bool g_af_ready = false;
bool g_sd_ready = false;

// ===== 計數（照片序號）=====
int imgCount = 0;

// ---------- LED ----------
static inline void led_init() {
  // Arduino-ESP32：ledcSetup(channel, freq, resolutionBits)
  ledcSetup(LEDC_LED_CH, LEDC_LED_FREQ, LEDC_LED_RES);
  ledcAttachPin(LED_PIN, LEDC_LED_CH);
  ledcWrite(LEDC_LED_CH, 0);
}

static inline void setLEDBrightness(uint8_t b) {
  ledcWrite(LEDC_LED_CH, b);
}

// ---------- SD 掛載 ----------
bool sd_begin_mhz(uint32_t mhz) {
  SdSpiConfig cfg(SD_CS, SHARED_SPI, SD_SCK_MHZ(mhz));
  if (sd.begin(cfg)) {
    g_sd_hz = mhz * 1000000UL;
    g_sd_ready = true;
    Serial.printf("[SD] mounted at %u MHz\n", (unsigned)mhz);
    return true;
  }
  return false;
}
bool sd_init_fast() {
  const uint32_t cand_mhz[] = {25, 20, 16, 12};
  for (size_t i=0;i<sizeof(cand_mhz)/sizeof(cand_mhz[0]);++i) {
    if (sd_begin_mhz(cand_mhz[i])) return true;
  }
  g_sd_ready = false;
  return false;
}

// ---------- 丟棄一幀 ----------
static inline void discard_one_frame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);
}

static bool autofocusReachedStableState(uint32_t timeout_ms) {
  const uint32_t start = millis();
  int stable_count = 0;

  while ((millis() - start) < timeout_ms) {
    const int status = ov5640_af_status();
    if (status == 0x10 || status == 0x70) {
      if (++stable_count >= AF_STABLE_COUNT) {
        return true;
      }
    } else if (status < 0) {
      return false;
    } else {
      stable_count = 0;
    }
    delay(AF_POLL_MS);
  }

  return false;
}

static void initAutofocus(sensor_t *s) {
#if ENABLE_AUTOFOCUS
  g_af_supported = ov5640_af_supported(s);
  if (!g_af_supported) {
    Serial.printf("[AF] unsupported sensor PID=0x%04X\n", s ? s->id.PID : 0);
    return;
  }

  Serial.printf("[AF] OV5640 AF library available: %s\n",
                ov5640_af_library_available() ? "yes" : "no");
  if (!ov5640_af_begin(s)) {
    Serial.printf("[AF] init failed, focusInit=%d\n", ov5640_af_init_result());
    Serial.println("[AF] Check whether the camera module actually has AF/VCM power.");
    return;
  }

  g_af_ready = true;
  Serial.printf("[AF] firmware ready, status=%d\n", ov5640_af_status());

  setLEDBrightness(LED_BRIGHTNESS);
  delay(LED_PREHEAT_MS);
  if (ov5640_af_enable_continuous()) {
    Serial.println("[AF] continuous autofocus enabled");
  } else {
    Serial.println("[AF] continuous autofocus start failed");
  }
  if (autofocusReachedStableState(AF_SETUP_SETTLE_MS)) {
    Serial.printf("[AF] stable focus reached, status=%d\n", ov5640_af_status());
  } else {
    Serial.printf("[AF] focus did not stabilize in time, status=%d\n", ov5640_af_status());
  }
  if (ov5640_af_stop()) {
    Serial.println("[AF] focus locked");
  } else {
    Serial.println("[AF] focus lock command failed");
  }
  setLEDBrightness(0);
#else
  (void)s;
#endif
}

// ---------- 分桶資料夾 ----------
static bool sd_ensure_dir(const char* path) {
  if (sd.exists(path)) return true;
  if (sd.mkdir(path)) {
    Serial.printf("[SD] mkdir: %s\n", path);
    return true;
  }
  Serial.printf("[SD] mkdir failed: %s\n", path);
  return false;
}

static bool remountSdCard() {
  sd.end();
  g_sd_ready = false;
  Serial.println("[SD] remounting...");
  if (!sd_init_fast()) {
    Serial.println("[SD] remount failed");
    return false;
  }
  return true;
}

static bool parseBucketNumber(const char *name, int *bucket) {
  if (!name || !bucket) return false;
  if (strlen(name) != 4 || name[0] != 'b') return false;
  if (!isdigit((unsigned char)name[1]) || !isdigit((unsigned char)name[2]) || !isdigit((unsigned char)name[3])) return false;
  *bucket = (name[1] - '0') * 100 + (name[2] - '0') * 10 + (name[3] - '0');
  return true;
}

static bool parseImageNumber(const char *name, int *number) {
  if (!name || !number) return false;
  if (strncmp(name, "image_", 6) != 0) return false;
  const char *ext = strstr(name, ".jpg");
  if (!ext || ext == name + 6 || ext[4] != '\0') return false;

  int value = 0;
  for (const char *p = name + 6; p < ext; ++p) {
    if (!isdigit((unsigned char)*p)) return false;
    value = value * 10 + (*p - '0');
  }
  *number = value;
  return true;
}

static void recoverImageCounter() {
  int last_bucket = -1;
  FsFile dir;

  if (!dir.open("/")) {
    Serial.println("[SD] root open failed, start counter at 0");
    imgCount = 0;
    return;
  }

  FsFile entry;
  while (entry.openNext(&dir, O_RDONLY)) {
    if (entry.isDir()) {
      char name[32];
      if (entry.getName(name, sizeof(name))) {
        int bucket = -1;
        if (parseBucketNumber(name, &bucket) && bucket > last_bucket) {
          last_bucket = bucket;
        }
      }
    }
    entry.close();
  }
  dir.close();

  if (last_bucket < 0) {
    imgCount = 0;
    Serial.println("[SD] no previous images found");
    return;
  }

  char folder[32];
  snprintf(folder, sizeof(folder), "/b%03d", last_bucket);

  int max_number = 0;
  if (!dir.open(folder)) {
    Serial.printf("[SD] open last bucket failed: %s\n", folder);
    imgCount = last_bucket * BUCKET_SIZE;
    return;
  }

  while (entry.openNext(&dir, O_RDONLY)) {
    if (entry.isFile()) {
      char name[64];
      if (entry.getName(name, sizeof(name))) {
        int number = 0;
        if (parseImageNumber(name, &number) && number > max_number) {
          max_number = number;
        }
      }
    }
    entry.close();
  }
  dir.close();

  imgCount = max_number;
  Serial.printf("[SD] recovered image counter: %d (next: %d)\n", imgCount, imgCount + 1);
}

// ---------- Atomic 寫入 + 驗證 ----------
bool atomicWriteAndVerify(const char* finalPath, const uint8_t* data, size_t len,
                          double* ms_out, double* mbps_out) {
  char tmpPath[96]; // 放大避免截斷
  snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", finalPath);

  File32 f = sd.open(tmpPath, O_CREAT | O_TRUNC | O_WRITE);
  if (!f) { Serial.println("[SD] open tmp failed"); return false; }

  // 預配置（失敗不致命）
  f.preAllocate(len);

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

  if (!f.sync()) {
    Serial.println("[SD] sync failed");
    f.close();
    sd.remove(tmpPath);
    return false;
  }
  f.close();

  if (sd.exists(finalPath) && !sd.remove(finalPath)) {
    Serial.println("[SD] remove final failed");
    sd.remove(tmpPath);
    return false;
  }

  if (!sd.rename(tmpPath, finalPath)) {
    Serial.println("[SD] rename tmp->final failed");
    sd.remove(tmpPath);
    return false;
  }

  File32 r = sd.open(finalPath, O_READ);
  if (!r) {
    Serial.println("[SD] reopen final failed");
    sd.remove(finalPath);
    return false;
  }
  uint64_t sz = r.size();
  r.close();

  if (sz != len) {
    Serial.printf("[SD] verify size mismatch: %llu vs %u\n",
                  (unsigned long long)sz, (unsigned)len);
    sd.remove(finalPath);
    return false;
  }

  int64_t t1 = esp_timer_get_time();
  double ms = (t1 - t0) / 1000.0;
  if (ms_out) *ms_out = ms;
  if (mbps_out) {
    *mbps_out = (len / (1024.0*1024.0)) / (ms / 1000.0);
  }
  return true;
}

// ---------- 曝光/增益設定 ----------
static void applyExposureMode(sensor_t *s) {
#if EXPOSURE_MODE == 0
  // ===== 自動曝光/增益 =====
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 1);

  s->set_exposure_ctrl(s, 1); // Auto exposure
  s->set_aec2(s, 0);
  s->set_ae_level(s, 0);

  s->set_gain_ctrl(s, 1);     // Auto gain
  s->set_gainceiling(s, (gainceiling_t)FIX_GAIN_CEIL); // 可保留天花板限制
  // aec_value/agc_gain 在 auto 下不保證生效（就不硬設）

#elif EXPOSURE_MODE == 1
  // ===== 固定曝光/固定增益（較一致）=====
  // 白平衡可選擇 auto（一般穩）或固定；這裡保留 auto WB
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 1);

  // 關閉自動曝光/增益
  s->set_exposure_ctrl(s, 0); // Manual exposure
  s->set_aec2(s, 0);
  s->set_ae_level(s, 0);
  s->set_aec_value(s, FIX_AEC_VALUE);

  s->set_gain_ctrl(s, 0);     // Manual gain
  s->set_agc_gain(s, FIX_AGC_GAIN);
  s->set_gainceiling(s, (gainceiling_t)FIX_GAIN_CEIL);
#endif

  // 其他畫質參數（保留你原本的設定）
  s->set_brightness(s, 0);     // -2 to 2
  s->set_contrast(s, 2);       // -2 to 2
  s->set_saturation(s, 1);     // -2 to 2
  s->set_special_effect(s, 0); // 0~6
  s->set_bpc(s, 0);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_dcw(s, 1);
  s->set_colorbar(s, 0);

#if EXPOSURE_MODE == 0
  Serial.println("[CAM] Exposure mode: AUTO");
#else
  Serial.printf("[CAM] Exposure mode: FIXED  AEC=%d  AGC=%d  CEIL=%d\n",
                FIX_AEC_VALUE, FIX_AGC_GAIN, FIX_GAIN_CEIL);
#endif
}

// ---------- 拍照 + 存檔（LED 先穩定，丟 2 幀再拍） ----------
void photo_save(const char *filePath) {
  if (!g_sd_ready && !remountSdCard()) {
    Serial.println("[SD] unavailable, skip capture save");
    return;
  }

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

  // 立刻把會用到的資料先存起來（避免 fb_return 後 use-after-free）
  const size_t img_len = fb->len;

  // 4) 關燈
  setLEDBrightness(0);
  int64_t t_led_off = esp_timer_get_time();

  // 5) 寫入（原子）+ 驗證；失敗降速重掛一次
  double write_ms = 0.0, write_mbps = 0.0;
  bool ok = atomicWriteAndVerify(filePath, fb->buf, img_len, &write_ms, &write_mbps);
  if (!ok) {
    Serial.println("[SD] write+verify failed, remount slower and retry once...");
    if (remountSdCard()) {
      ok = atomicWriteAndVerify(filePath, fb->buf, img_len, &write_ms, &write_mbps);
    }
    if (!ok) {
      g_sd_ready = false;
    }
  }

  // 6) 釋放影像緩衝
  esp_camera_fb_return(fb);
  int64_t t_after_fb_return = esp_timer_get_time();

  // 7) 時間資訊
  (void)t_after_fb_return;
  double led_to_fb_ms    = (t_got_fb - t_led_on) / 1000.0;
  double fb_to_ledoff_ms = (t_led_off - t_got_fb) / 1000.0;

  Serial.printf("Timings  LED->FB: %.1f ms, FB->LEDoff: %.1f ms, FB->SD(write): %.1f ms\n",
                led_to_fb_ms, fb_to_ledoff_ms, write_ms);
  Serial.printf("Size: %.2f MB, SD: %.2f MB/s  (SPI %.0f MHz)\n",
                img_len / (1024.0 * 1024.0), write_mbps, g_sd_hz / 1e6);

  if (ok) Serial.printf("[OK] Saved: %s\n", filePath);
  else    Serial.printf("[FAIL] Save failed: %s\n", filePath);

  delay(10);
}

void setup() {
  Serial.begin(115200);
  delay(100);

  SPI.begin();

  // LED：用 LEDC（避免與 camera LEDC 衝突）
  pinMode(LED_PIN, OUTPUT);
  led_init();

  // ===== 相機設定（單緩衝 + WHEN_EMPTY 是關鍵）=====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; // camera 用 CH0/T0
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = XCLK_FREQ_HZ;
  config.frame_size   = FRAMESIZE_QSXGA;      // 需要更穩就先降 UXGA/SXGA
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count     = 1;                    // ⭐ 單緩衝

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[CAM] init failed");
    while (true) { delay(1000); }
  }

  // 套用曝光/增益模式
  sensor_t *s = esp_camera_sensor_get();
  applyExposureMode(s);
  initAutofocus(s);

  // 掛載 SD（多時脈嘗試）
  if (!sd_init_fast()) {
    Serial.println("[SD] mount failed (all speeds tried).");
    while (true) { delay(1000); }
  }

  recoverImageCounter();

  Serial.println("Continuous mode: capture (no sleep).");
}

void loop() {
  static const uint32_t period_ms = TARGET_PERIOD_S * 1000UL;

  // 用「絕對時間」排程：初始化 next_tick
  static uint32_t next_tick = 0;
  if (next_tick == 0) next_tick = millis() + period_ms;

  ++imgCount;

  // ---- 分桶：/bXXX ----
  int bucket = (imgCount - 1) / BUCKET_SIZE;
  char folder[32];
  snprintf(folder, sizeof(folder), "/b%03d", bucket);

  static int last_bucket = -1;
  if (bucket != last_bucket) {
    sd_ensure_dir(folder);
    last_bucket = bucket;
  }

  // 檔名
  char filename[96];
  snprintf(filename, sizeof(filename), "%s/image_%d.jpg", folder, imgCount);

  photo_save(filename);

  // ---- 對齊下一個邊界 ----
  // 先推進下一個 tick
  next_tick += period_ms;

  // delay 到 next_tick（若還沒到）
  int32_t remain = (int32_t)(next_tick - millis());
  if (remain > 0) delay((uint32_t)remain);

  // ---- 追相位：只有「真的落後 >= 1 個週期」才 skip ----
  // 加一個小容忍（例如 20ms），避免 delay 醒來晚 1~2ms 就誤判
  const int32_t SLACK_MS = 20;
  int32_t behind = (int32_t)(millis() - next_tick);

  if (behind > SLACK_MS) {
    if (behind >= (int32_t)period_ms) {
      uint32_t skip = (behind / (int32_t)period_ms); // 已經落後幾個完整週期
      next_tick += skip * period_ms;
      Serial.printf("[SCHED] behind %ld ms, skip %u slot(s) to realign.\n",
                    (long)behind, (unsigned)skip);
    } else {
      // 只是小落後（< 1 週期），不 skip，只記錄
      Serial.printf("[SCHED] minor slip %ld ms (no skip)\n", (long)behind);
    }
  }
}

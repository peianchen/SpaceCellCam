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
#include "esp_heap_caps.h"
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <SdFat.h>
#include "ov5640_af_support.h"

#if __has_include("driver/temp_sensor.h")
#include "driver/temp_sensor.h"
#define HAS_LEGACY_TEMP_SENSOR 1
#else
#define HAS_LEGACY_TEMP_SENSOR 0
#endif

// ===== 相機腳位（Seeed XIAO ESP32S3）=====
#ifndef CAMERA_MODEL_XIAO_ESP32S3
#define CAMERA_MODEL_XIAO_ESP32S3
#endif
#include "camera_pins.h"

// ===== 參數 =====
#define TARGET_PERIOD_S   10         // 週期（秒）
#define LED_PIN           D1         // 補光 LED 腳位
#define DEBUG_UART_RX_PIN D7         // XIAO ESP32S3 RX / GPIO44，接 Debug Mate TX
#define DEBUG_UART_TX_PIN D6         // XIAO ESP32S3 TX / GPIO43，接 Debug Mate RX
#define DEBUG_BAUD        115200
#define SD_CS             21         // SD CS 腳位
#define BUCKET_SIZE       500        // 每個資料夾最多檔案數
#define XCLK_FREQ_HZ      20000000   // OV5640 常用 20 MHz；若仍出現壞圖再降 10 MHz
#define CAPTURE_FRAME_SIZE FRAMESIZE_QSXGA
#define CAPTURE_RESTORE_SETTLE_MS 50 // AF 從 QVGA 切回 QSXGA 後，先讓 sensor/JPEG 穩定
#define CAPTURE_DISCARD_FRAMES 1     // 正式拍照前丟幾幀；2 較穩但 LED 會多亮約 1~3 秒
#define LED_PREHEAT_MS    300        // LED 預熱
#define LED_BRIGHTNESS    80         // LED 亮度（0~255）
#define STATUS_LED_BRIGHTNESS 100     // 不接Serial時用補光LED顯示狀態
#define ENABLE_DEBUG_LOG  1          // 1=印出詳細 debug log；0=只保留必要訊息
#define ENABLE_DEBUG_MATE_LOG 1      // 1=Serial1/Debug Mate 顯示短狀態；USB CDC 仍顯示完整 log
#define ENABLE_CAMERA_LIFECYCLE_DEBUG 1 // 1=記錄 fb_get/return、記憶體、溫度、standby 狀態
#define ENABLE_FB_FRAME_LOG 0        // 1=每次 fb_get/return 都印；0=只保留 runtime count，避免 AF pump 洗版
#define ENABLE_SENSOR_STANDBY_AFTER_CAPTURE 1 // 1=每次拍照後嘗試讓 OV5640 進入 standby
#define COPY_FRAME_TO_PSRAM_BEFORE_SD 1 // 1=先複製 JPEG 並 fb_return/standby，再寫 SD，降低 sensor 工作時間
#define SENSOR_STANDBY_SETTLE_MS 80  // 進入 standby 後等待時間
#define SENSOR_WAKE_SETTLE_MS 120    // 從 standby 喚醒後等待時間
#define AF_DEBUG_WAIT_LOG 0          // 1=印 AF wait/pump 細節；0=關閉，避免 Serial 輸出拖慢 AF
#define AF_DEBUG_EVERY_POLL 0        // 1=每次 AF 輪詢都印 status，只有 AF_DEBUG_WAIT_LOG=1 時有效
#define ENABLE_AUTOFOCUS  1          // 只在 OV5640 + AF 模組時生效
#define AF_LOCK_MODE      3          // 0=關AF  1=setup後stop鎖焦  2=setup後release  3=每張中心single AF後release
#define AF_LED_PREHEAT_MS 50         // AF 補光預熱，獨立於正式拍照補光
#define AF_LED_BRIGHTNESS 24         // AF 補光亮度，降低對樣本的影響
#define AF_SETUP_SETTLE_MS 800       // setup single AF 最多等待時間
#define AF_CAPTURE_SETTLE_MS 2500    // 每張 AF 最多等待時間；AF pump 先降 QVGA
#define AF_PUMP_FRAME_SIZE FRAMESIZE_QVGA
#define AF_SWITCH_FRAME_SIZE_FOR_PUMP 1 // 1=AF pump 先切 QVGA，加快對焦；拍照前切回 QSXGA
#define AF_PUMP_FRAME_COUNT 5        // AF 時快速出幾張低解析度 frame 讓韌體判斷
#define AF_KEEP_LOCK_AFTER_CAPTURE 0 // 0=每張拍完 release，下張重新對焦；1=拍完不 release
#define AF_REFOCUS_EVERY_N 0         // 0=成功後不週期重對焦；>0=每 N 張重對焦一次
#define AF_POLL_MS        25         // AF 狀態輪詢間隔
#define AF_STABLE_COUNT   1          // 連續幾次 focused 才視為已對焦
/*#define AF_CENTER_X       30         // OV5640 AF grid: 0~79
#define AF_CENTER_Y       22         // OV5640 AF grid: 0~59
#define AF_CENTER_W       20
#define AF_CENTER_H       168*/
#define AF_CENTER_X 34
#define AF_CENTER_Y 24
#define AF_CENTER_W 12
#define AF_CENTER_H 10
#define AWB_WARMUP_FRAMES 8          // 開機時在LED光源下讓AWB收斂的幀數
#define WB_MODE           0          // 0=Auto  1=Sunny  2=Cloudy  3=Office  4=Home
#define ENABLE_AWB_LOCK   0          // 0=持續自動白平衡；1=warmup後鎖定

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
bool g_af_focus_locked = false;
int g_af_last_focus_img = 0;
bool g_sd_ready = false;
bool g_sensor_standby = false;
uint32_t g_fb_get_count = 0;
uint32_t g_fb_return_count = 0;
uint32_t g_capture_count = 0;
uint32_t g_last_capture_ms = 0;
uint32_t g_last_fb_get_ms = 0;
uint32_t g_last_fb_return_ms = 0;
const bool g_mjpeg_stream_active = false;

// ===== 計數（照片序號）=====
int imgCount = 0;

struct DualLogger {
  void begin(unsigned long baud) {
    Serial.begin(baud);
    Serial1.begin(baud, SERIAL_8N1, DEBUG_UART_RX_PIN, DEBUG_UART_TX_PIN);
  }

  template <typename T>
  void print(const T &value) {
    Serial.print(value);
  }

  template <typename T>
  void println(const T &value) {
    Serial.println(value);
  }

  void println() {
    Serial.println();
  }

  int printf(const char *format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    const int written = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    Serial.print(buffer);
    return written;
  }

  template <typename T>
  void matePrintln(const T &value) {
    Serial1.println(value);
  }

  int matePrintf(const char *format, ...) {
    char buffer[96];
    va_list args;
    va_start(args, format);
    const int written = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    Serial1.print(buffer);
    return written;
  }
};

DualLogger LOG;

#if ENABLE_DEBUG_LOG
#define DBG_PRINTF(...) LOG.printf(__VA_ARGS__)
#define DBG_PRINTLN(x) LOG.println(x)
#else
#define DBG_PRINTF(...)
#define DBG_PRINTLN(x)
#endif

#if ENABLE_DEBUG_MATE_LOG
#define DM_PRINTF(...) LOG.matePrintf(__VA_ARGS__)
#define DM_PRINTLN(x) LOG.matePrintln(x)
#else
#define DM_PRINTF(...)
#define DM_PRINTLN(x)
#endif

static const char* afStatusName(int status) {
  switch (status) {
    case 0x10: return "focused";
    case 0x70: return "idle";
    case -1: return "not-ready";
    case -2: return "fw-not-ready";
    case -3: return "unsupported";
    default:
      if (status >= 0x00 && status <= 0x0F) return "busy";
      if (status >= 0x80 && status <= 0x8F) return "searching";
      return "unknown";
  }
}

static float readCpuTempC() {
#if HAS_LEGACY_TEMP_SENSOR
  static bool initialized = false;
  if (!initialized) {
    temp_sensor_config_t cfg = TSENS_CONFIG_DEFAULT();
    cfg.dac_offset = TSENS_DAC_L2;
    if (temp_sensor_set_config(cfg) != ESP_OK) {
      return NAN;
    }
    if (temp_sensor_start() != ESP_OK) {
      return NAN;
    }
    initialized = true;
  }
  float temp_c = NAN;
  if (temp_sensor_read_celsius(&temp_c) != ESP_OK) {
    return NAN;
  }
  return temp_c;
#else
  return NAN;
#endif
}

static void logRuntimeState(const char *tag) {
#if ENABLE_CAMERA_LIFECYCLE_DEBUG
  const float temp_c = readCpuTempC();
  LOG.printf("[RUNTIME] %s uptime=%lu ms captures=%lu fb_get=%lu fb_return=%lu "
             "last_get=%lu ms last_return=%lu ms mjpeg_stream=%d standby=%d "
             "free_heap=%u free_psram=%u cpu_temp=",
             tag,
             (unsigned long)millis(),
             (unsigned long)g_capture_count,
             (unsigned long)g_fb_get_count,
             (unsigned long)g_fb_return_count,
             (unsigned long)g_last_fb_get_ms,
             (unsigned long)g_last_fb_return_ms,
             g_mjpeg_stream_active ? 1 : 0,
             g_sensor_standby ? 1 : 0,
             (unsigned)ESP.getFreeHeap(),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  if (isnan(temp_c)) {
    LOG.println("n/a");
  } else {
    LOG.printf("%.1f C\n", temp_c);
  }
#else
  (void)tag;
#endif
}

static bool setSensorStandby(bool standby) {
#if ENABLE_SENSOR_STANDBY_AFTER_CAPTURE
  sensor_t *s = esp_camera_sensor_get();
  if (!s) {
    LOG.printf("[CAM][STANDBY] %s failed: no sensor\n", standby ? "enter" : "wake");
    return false;
  }

  const uint8_t value = standby ? 0x42 : 0x02;
  const int rc = s->set_reg(s, 0x3008, 0xFF, value);
  if (rc == 0) {
    g_sensor_standby = standby;
    LOG.printf("[CAM][STANDBY] %s ok reg=0x3008 value=0x%02X\n",
               standby ? "enter" : "wake", value);
    DM_PRINTLN(standby ? "STBY ON" : "WAKE");
    delay(standby ? SENSOR_STANDBY_SETTLE_MS : SENSOR_WAKE_SETTLE_MS);
    logRuntimeState(standby ? "standby_entered" : "standby_woke");
    return true;
  }

  LOG.printf("[CAM][STANDBY] %s failed rc=%d reg=0x3008 value=0x%02X\n",
             standby ? "enter" : "wake", rc, value);
  return false;
#else
  (void)standby;
  return false;
#endif
}

static void wakeSensorIfNeeded(const char *reason) {
#if ENABLE_SENSOR_STANDBY_AFTER_CAPTURE
  if (g_sensor_standby) {
    LOG.printf("[CAM][STANDBY] wake requested reason=%s\n", reason);
    setSensorStandby(false);
  }
#else
  (void)reason;
#endif
}

static camera_fb_t *cameraFbGetLogged(const char *tag) {
  const int64_t start = esp_timer_get_time();
  camera_fb_t *fb = esp_camera_fb_get();
  const int64_t end = esp_timer_get_time();
  g_last_fb_get_ms = (uint32_t)((end - start) / 1000);
  ++g_fb_get_count;
#if ENABLE_CAMERA_LIFECYCLE_DEBUG && ENABLE_FB_FRAME_LOG
  LOG.printf("[FB][GET] tag=%s count=%lu elapsed=%.1f ms fb=%p len=%u format=%d "
             "heap=%u psram=%u standby=%d mjpeg_stream=%d\n",
             tag,
             (unsigned long)g_fb_get_count,
             (end - start) / 1000.0,
             fb,
             fb ? (unsigned)fb->len : 0,
             fb ? fb->format : -1,
             (unsigned)ESP.getFreeHeap(),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             g_sensor_standby ? 1 : 0,
             g_mjpeg_stream_active ? 1 : 0);
#else
  (void)tag;
#endif
  return fb;
}

static void cameraFbReturnLogged(camera_fb_t *fb, const char *tag) {
  if (!fb) return;
  const int64_t start = esp_timer_get_time();
  esp_camera_fb_return(fb);
  const int64_t end = esp_timer_get_time();
  g_last_fb_return_ms = (uint32_t)((end - start) / 1000);
  ++g_fb_return_count;
#if ENABLE_CAMERA_LIFECYCLE_DEBUG && ENABLE_FB_FRAME_LOG
  LOG.printf("[FB][RETURN] tag=%s count=%lu elapsed=%.1f ms fb=%p heap=%u psram=%u standby=%d\n",
             tag,
             (unsigned long)g_fb_return_count,
             (end - start) / 1000.0,
             fb,
             (unsigned)ESP.getFreeHeap(),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             g_sensor_standby ? 1 : 0);
#else
  (void)tag;
#endif
}

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

static void statusBlink(uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    setLEDBrightness(STATUS_LED_BRIGHTNESS);
    delay(120);
    setLEDBrightness(0);
    delay(120);
  }
}

static void fatalBlinkLoop(uint8_t count) {
  while (true) {
    statusBlink(count);
    delay(1000);
  }
}

// ---------- SD 掛載 ----------
bool sd_begin_mhz(uint32_t mhz) {
  SdSpiConfig cfg(SD_CS, SHARED_SPI, SD_SCK_MHZ(mhz));
  if (sd.begin(cfg)) {
    g_sd_hz = mhz * 1000000UL;
    g_sd_ready = true;
    LOG.printf("[SD] mounted at %u MHz\n", (unsigned)mhz);
    DM_PRINTF("SD %uM OK\n", (unsigned)mhz);
    return true;
  }
  return false;
}
bool sd_init_fast() {
  const uint32_t cand_mhz[] = {40, 32, 25, 20, 16, 12};
  for (size_t i=0;i<sizeof(cand_mhz)/sizeof(cand_mhz[0]);++i) {
    if (sd_begin_mhz(cand_mhz[i])) return true;
  }
  g_sd_ready = false;
  return false;
}

// ---------- 丟棄一幀 ----------
static inline void discard_one_frame() {
  camera_fb_t *fb = cameraFbGetLogged("discard");
  cameraFbReturnLogged(fb, "discard");
}

static bool autofocusReachedStableState(uint32_t timeout_ms, bool require_focused, bool pump_frames, uint8_t min_pump_frames, bool accept_after_min_pump) {
  const uint32_t start = millis();
  int stable_count = 0;
  int last_status = 9999;
  uint8_t pumped_frames = 0;

#if AF_DEBUG_WAIT_LOG
  DBG_PRINTF("[AF][WAIT] start timeout=%lu ms require_focused=%d stable_count=%d pump_frames=%d min_pump=%u\n",
             (unsigned long)timeout_ms, require_focused ? 1 : 0, AF_STABLE_COUNT, pump_frames ? 1 : 0, min_pump_frames);
#endif

  while ((millis() - start) < timeout_ms) {
    const int status = ov5640_af_status();
    const uint32_t elapsed = millis() - start;
    const bool focused = status == 0x10;
    const bool stable_idle = !require_focused && status == 0x70;
#if AF_DEBUG_WAIT_LOG
#if AF_DEBUG_EVERY_POLL
    DBG_PRINTF("[AF][WAIT] t=%lu ms status=0x%02X(%s) stable=%d/%d\n",
               (unsigned long)elapsed, status, afStatusName(status), stable_count, AF_STABLE_COUNT);
#else
    if (status != last_status) {
      DBG_PRINTF("[AF][WAIT] t=%lu ms status=0x%02X(%s) stable=%d/%d\n",
                 (unsigned long)elapsed, status, afStatusName(status), stable_count, AF_STABLE_COUNT);
      last_status = status;
    }
#endif
#endif
    if ((focused || stable_idle) && pumped_frames >= min_pump_frames) {
      if (++stable_count >= AF_STABLE_COUNT) {
#if AF_DEBUG_WAIT_LOG
        DBG_PRINTF("[AF][WAIT] done t=%lu ms status=0x%02X(%s)\n",
                   (unsigned long)(millis() - start), status, afStatusName(status));
#endif
        return true;
      }
    } else if (status < 0) {
#if AF_DEBUG_WAIT_LOG
      DBG_PRINTF("[AF][WAIT] abort t=%lu ms status=%d(%s)\n",
                 (unsigned long)elapsed, status, afStatusName(status));
#endif
      return false;
    } else {
      stable_count = 0;
    }

    if (pump_frames) {
      camera_fb_t *fb = cameraFbGetLogged("af_pump");
      if (fb) {
        const size_t pump_len = fb->len;
        cameraFbReturnLogged(fb, "af_pump");
        ++pumped_frames;
#if AF_DEBUG_WAIT_LOG
        DBG_PRINTF("[AF][WAIT] pump frame len=%u\n", (unsigned)pump_len);
#endif
      } else {
#if AF_DEBUG_WAIT_LOG
        DBG_PRINTLN("[AF][WAIT] pump frame failed");
#endif
      }
    }
    delay(AF_POLL_MS);
  }

  const int final_status = ov5640_af_status();
#if AF_DEBUG_WAIT_LOG
  DBG_PRINTF("[AF][WAIT] timeout t=%lu ms final_status=0x%02X(%s)\n",
             (unsigned long)(millis() - start), final_status, afStatusName(final_status));
#endif
  if ((final_status == 0x10 || (!require_focused && final_status == 0x70)) &&
      pumped_frames >= min_pump_frames) {
    return true;
  }
  return false;
}

static void initAutofocus(sensor_t *s) {
#if ENABLE_AUTOFOCUS && (AF_LOCK_MODE != 0)
  const int64_t t_af_start = esp_timer_get_time();
  LOG.printf("[AF] config lock_mode=%d setup_timeout=%d ms capture_timeout=%d ms poll=%d ms stable=%d led=%d preheat=%d ms\n",
                AF_LOCK_MODE, AF_SETUP_SETTLE_MS, AF_CAPTURE_SETTLE_MS, AF_POLL_MS, AF_STABLE_COUNT,
                AF_LED_BRIGHTNESS, AF_LED_PREHEAT_MS);
  g_af_supported = ov5640_af_supported(s);
  if (!g_af_supported) {
    LOG.printf("[AF] unsupported sensor PID=0x%04X\n", s ? s->id.PID : 0);
    return;
  }

  LOG.printf("[AF] OV5640 AF library available: %s\n",
                ov5640_af_library_available() ? "yes" : "no");
  if (!ov5640_af_begin(s)) {
    LOG.printf("[AF] init failed, focusInit=%d\n", ov5640_af_init_result());
    LOG.println("[AF] Check whether the camera module actually has AF/VCM power.");
    return;
  }

  g_af_ready = true;
  LOG.printf("[AF] firmware ready, status=%d(%s)\n", ov5640_af_status(), afStatusName(ov5640_af_status()));

#if AF_LOCK_MODE == 3
  LOG.println("[AF] per-capture autofocus enabled");
  if (ov5640_af_set_focus_zone(AF_CENTER_X, AF_CENTER_Y, AF_CENTER_W, AF_CENTER_H)) {
    LOG.printf("[AF] center focus zone x=%d y=%d w=%d h=%d\n",
                  AF_CENTER_X, AF_CENTER_Y, AF_CENTER_W, AF_CENTER_H);
  } else {
    LOG.println("[AF] center focus zone setup failed");
  }
#else
  DBG_PRINTF("[AF][LED] on brightness=%d preheat=%d ms\n", AF_LED_BRIGHTNESS, AF_LED_PREHEAT_MS);
  setLEDBrightness(AF_LED_BRIGHTNESS);
  delay(AF_LED_PREHEAT_MS);

  if (ov5640_af_set_focus_zone(AF_CENTER_X, AF_CENTER_Y, AF_CENTER_W, AF_CENTER_H)) {
    LOG.printf("[AF] center focus zone x=%d y=%d w=%d h=%d\n",
                  AF_CENTER_X, AF_CENTER_Y, AF_CENTER_W, AF_CENTER_H);
  } else {
    LOG.println("[AF] center focus zone setup failed");
  }

  if (ov5640_af_trigger_single()) {
    LOG.printf("[AF] setup single autofocus triggered, status=%d(%s)\n", ov5640_af_status(), afStatusName(ov5640_af_status()));
  } else {
    LOG.println("[AF] setup single autofocus start failed");
  }
  if (autofocusReachedStableState(AF_SETUP_SETTLE_MS, true, true, 1, false)) {
    LOG.printf("[AF] stable focus reached, status=%d\n", ov5640_af_status());
#if AF_LOCK_MODE == 2
    if (ov5640_af_release()) {
      LOG.println("[AF] focus released (VCM heat test mode)");
    } else {
      LOG.println("[AF] focus release command failed");
    }
#else
    if (ov5640_af_stop()) {
      LOG.println("[AF] focus locked");
    } else {
      LOG.println("[AF] focus lock command failed");
    }
#endif
  } else {
    LOG.printf("[AF] WARNING: focus did not stabilize in time, status=%d(%s) - not locking\n",
                  ov5640_af_status(), afStatusName(ov5640_af_status()));
  }
  setLEDBrightness(0);
  DBG_PRINTF("[AF][LED] off, setup_af_elapsed=%.1f ms\n", (esp_timer_get_time() - t_af_start) / 1000.0);
#endif
#else
  (void)s;
  LOG.println("[AF] disabled by config");
#endif
}

static bool runCaptureAutofocus() {
#if ENABLE_AUTOFOCUS && (AF_LOCK_MODE == 3)
  const int64_t t_af_start = esp_timer_get_time();
  if (!g_af_ready) return false;
  sensor_t *s = esp_camera_sensor_get();

#if AF_KEEP_LOCK_AFTER_CAPTURE
  if (g_af_focus_locked && (AF_REFOCUS_EVERY_N <= 0 || (imgCount - g_af_last_focus_img) < AF_REFOCUS_EVERY_N)) {
    DBG_PRINTF("[AF][CAP] reuse locked focus from img=%d, current=%d\n", g_af_last_focus_img, imgCount);
    return true;
  }
#endif

  DBG_PRINTF("[AF][CAP] start tap-style center focus, status_before=%d(%s)\n", ov5640_af_status(), afStatusName(ov5640_af_status()));
#if AF_SWITCH_FRAME_SIZE_FOR_PUMP
  if (s && s->set_framesize(s, AF_PUMP_FRAME_SIZE) == 0) {
    DBG_PRINTLN("[AF][CAP] frame size switched for fast AF pump");
  } else {
    LOG.println("[AF] WARNING: failed to switch frame size for AF pump");
  }
#else
  (void)s;
  DBG_PRINTLN("[AF][CAP] keeping QSXGA frame size for AF pump");
#endif
  if (!ov5640_af_focus_zone(AF_CENTER_X, AF_CENTER_Y, AF_CENTER_W, AF_CENTER_H)) {
    LOG.println("[AF] per-capture start failed");
#if AF_SWITCH_FRAME_SIZE_FOR_PUMP
    if (s) s->set_framesize(s, CAPTURE_FRAME_SIZE);
#endif
    return false;
  }
  if (!autofocusReachedStableState(AF_CAPTURE_SETTLE_MS, false, true, AF_PUMP_FRAME_COUNT, false)) {
    LOG.printf("[AF] per-capture did not stabilize, status=%d\n", ov5640_af_status());
#if AF_SWITCH_FRAME_SIZE_FOR_PUMP
    if (s) s->set_framesize(s, CAPTURE_FRAME_SIZE);
#endif
    return false;
  }
  if (ov5640_af_stop()) {
    const double af_elapsed_ms = (esp_timer_get_time() - t_af_start) / 1000.0;
    LOG.printf("[AF] per-capture focus confirmed, elapsed=%.1f ms, status=%d(%s)\n",
                  af_elapsed_ms, ov5640_af_status(), afStatusName(ov5640_af_status()));
    DM_PRINTF("AF %.0fms\n", af_elapsed_ms);
    g_af_focus_locked = true;
    g_af_last_focus_img = imgCount;
  }
#if AF_SWITCH_FRAME_SIZE_FOR_PUMP
  if (s && s->set_framesize(s, CAPTURE_FRAME_SIZE) == 0) {
    DBG_PRINTLN("[AF][CAP] frame size restored for QSXGA capture");
    delay(CAPTURE_RESTORE_SETTLE_MS);
  } else {
    LOG.println("[AF] WARNING: failed to restore capture frame size");
  }
#endif
  return true;
#else
  return false;
#endif
}

static void releaseCaptureAutofocus() {
#if ENABLE_AUTOFOCUS && (AF_LOCK_MODE == 3)
#if AF_KEEP_LOCK_AFTER_CAPTURE
  if (g_af_focus_locked) {
    DBG_PRINTLN("[AF] keep focus locked after capture");
    return;
  }
#endif
  if (g_af_ready && ov5640_af_release()) {
    LOG.println("[AF] per-capture released");
  }
#endif
}

// ---------- 分桶資料夾 ----------
static bool sd_ensure_dir(const char* path) {
  if (sd.exists(path)) return true;
  if (sd.mkdir(path)) {
    LOG.printf("[SD] mkdir: %s\n", path);
    return true;
  }
  LOG.printf("[SD] mkdir failed: %s\n", path);
  return false;
}

static bool remountSdCard() {
  sd.end();
  g_sd_ready = false;
  LOG.println("[SD] remounting...");
  if (!sd_init_fast()) {
    LOG.println("[SD] remount failed");
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
  File32 dir = sd.open("/", O_RDONLY);

  if (!dir || !dir.isDir()) {
    LOG.println("[SD] root open failed, start counter at 0");
    imgCount = 0;
    return;
  }

  File32 entry;
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
    LOG.println("[SD] no previous images found");
    return;
  }

  char folder[32];
  snprintf(folder, sizeof(folder), "/b%03d", last_bucket);

  int max_number = 0;
  dir = sd.open(folder, O_RDONLY);
  if (!dir || !dir.isDir()) {
    LOG.printf("[SD] open last bucket failed: %s\n", folder);
    imgCount = (last_bucket + 1) * BUCKET_SIZE;
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

  imgCount = (max_number == 0) ? (last_bucket * BUCKET_SIZE) : max_number;
  LOG.printf("[SD] recovered image counter: %d (next: %d)\n", imgCount, imgCount + 1);
}

// ---------- Atomic 寫入 + 驗證 ----------
bool atomicWriteAndVerify(const char* finalPath, const uint8_t* data, size_t len,
                          double* ms_out, double* mbps_out) {
  char tmpPath[96]; // 放大避免截斷
  snprintf(tmpPath, sizeof(tmpPath), "%s.tmp", finalPath);

  File32 f = sd.open(tmpPath, O_CREAT | O_TRUNC | O_WRITE);
  if (!f) { LOG.println("[SD] open tmp failed"); return false; }

  // 預配置（失敗不致命）
  f.preAllocate(len);

  int64_t t0 = esp_timer_get_time();
  const size_t CHUNK = 32 * 1024;
  size_t written = 0;

  while (written < len) {
    size_t n = (len - written) > CHUNK ? CHUNK : (len - written);
    size_t w = f.write(data + written, n);
    if (w != n) {
      LOG.printf("[SD] write short (%u/%u) at %u\n",
                    (unsigned)w, (unsigned)n, (unsigned)written);
      f.close();
      sd.remove(tmpPath);
      return false;
    }
    written += n;
  }

  if (!f.sync()) {
    LOG.println("[SD] sync failed");
    f.close();
    sd.remove(tmpPath);
    return false;
  }
  f.close();

  if (sd.exists(finalPath) && !sd.remove(finalPath)) {
    LOG.println("[SD] remove final failed");
    sd.remove(tmpPath);
    return false;
  }

  if (!sd.rename(tmpPath, finalPath)) {
    LOG.println("[SD] rename tmp->final failed");
    sd.remove(tmpPath);
    return false;
  }

  File32 r = sd.open(finalPath, O_READ);
  if (!r) {
    LOG.println("[SD] reopen final failed");
    sd.remove(finalPath);
    return false;
  }
  uint64_t sz = r.size();
  r.close();

  if (sz != len) {
    LOG.printf("[SD] verify size mismatch: %llu vs %u\n",
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
  s->set_wb_mode(s, WB_MODE);

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
  s->set_wb_mode(s, WB_MODE);

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
  s->set_contrast(s, 0);       // -2 to 2
  s->set_saturation(s, 0);     // -2 to 2
  s->set_special_effect(s, 0); // 0~6
  s->set_bpc(s, 0);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_dcw(s, 1);
  s->set_colorbar(s, 0);

#if EXPOSURE_MODE == 0
  LOG.println("[CAM] Exposure mode: AUTO");
#else
  LOG.printf("[CAM] Exposure mode: FIXED  AEC=%d  AGC=%d  CEIL=%d\n",
                FIX_AEC_VALUE, FIX_AGC_GAIN, FIX_GAIN_CEIL);
#endif
}

static void warmupAndLockWhiteBalance(sensor_t *s) {
  if (!s) return;

  LOG.printf("[CAM] AWB warmup under LED: %d frames\n", AWB_WARMUP_FRAMES);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, WB_MODE);

  setLEDBrightness(LED_BRIGHTNESS);
  delay(LED_PREHEAT_MS);
  for (int i = 0; i < AWB_WARMUP_FRAMES; ++i) {
    discard_one_frame();
    delay(30);
  }

#if ENABLE_AWB_LOCK
  s->set_whitebal(s, 0);
  LOG.println("[CAM] AWB locked after LED warmup");
#else
  LOG.println("[CAM] AWB left automatic after LED warmup");
#endif
  setLEDBrightness(0);
}

// ---------- 拍照 + 存檔（LED 先穩定，丟 2 幀再拍） ----------
void photo_save(const char *filePath) {
  const int64_t t_photo_start = esp_timer_get_time();
  const char *file_name = strrchr(filePath, '/');
  file_name = file_name ? file_name + 1 : filePath;
  ++g_capture_count;
  const uint32_t now_ms = millis();
  const uint32_t interval_ms = g_last_capture_ms ? (now_ms - g_last_capture_ms) : 0;
  g_last_capture_ms = now_ms;
  LOG.printf("[CAP][INTERVAL] capture=%lu img=%d interval=%lu ms target=%lu ms\n",
             (unsigned long)g_capture_count, imgCount,
             (unsigned long)interval_ms, (unsigned long)(TARGET_PERIOD_S * 1000UL));
  DM_PRINTF("CAP %s\n", file_name);
  DM_PRINTF("INT %.1fs\n", interval_ms / 1000.0);
  logRuntimeState("before_capture");
  wakeSensorIfNeeded("capture_start");

  DBG_PRINTF("[CAP] begin img=%d path=%s sd_ready=%d af_ready=%d af_mode=%d\n",
             imgCount, filePath, g_sd_ready ? 1 : 0, g_af_ready ? 1 : 0, AF_LOCK_MODE);
  if (!g_sd_ready && !remountSdCard()) {
    LOG.println("[SD] unavailable, skip capture save");
    DM_PRINTLN("SD FAIL");
    setSensorStandby(true);
    return;
  }

  int64_t t_led_on = esp_timer_get_time();

  // 1) LED 預熱與穩定
  DBG_PRINTF("[CAP][LED] on brightness=%d preheat=%d ms\n", LED_BRIGHTNESS, LED_PREHEAT_MS);
  setLEDBrightness(LED_BRIGHTNESS);
  delay(LED_PREHEAT_MS);
#if ENABLE_AUTOFOCUS && (AF_LOCK_MODE == 3)
  if (!runCaptureAutofocus()) {
    LOG.println("[AF] capture skipped because focus was not confirmed");
    DM_PRINTLN("AF FAIL");
    setLEDBrightness(0);
    releaseCaptureAutofocus();
    setSensorStandby(true);
    logRuntimeState("capture_skipped_af");
    return;
  }
#else
  runCaptureAutofocus();
#endif

  // 2) 丟幀（確保下一幀的曝光起點在 LED 穩定後）
  const int64_t t_discard_start = esp_timer_get_time();
  for (uint8_t i = 0; i < CAPTURE_DISCARD_FRAMES; ++i) {
    discard_one_frame();
  }
  const int64_t t_discard_end = esp_timer_get_time();
  DBG_PRINTF("[CAP] discarded %d frames in %.1f ms\n",
             CAPTURE_DISCARD_FRAMES, (t_discard_end - t_discard_start) / 1000.0);

  // 3) 正式取幀
  camera_fb_t *fb = cameraFbGetLogged("final");
  int64_t t_got_fb = esp_timer_get_time();
  if (!fb) {
    LOG.println("[CAM] esp_camera_fb_get() failed");
    DM_PRINTLN("FB FAIL");
    setLEDBrightness(0);
    releaseCaptureAutofocus();
    setSensorStandby(true);
    logRuntimeState("capture_failed_fb_get");
    return;
  }

  // 立刻把會用到的資料先存起來（避免 fb_return 後 use-after-free）
  const size_t img_len = fb->len;
  DBG_PRINTF("[CAP] fb_get=%lu ms len=%u bytes format=%d\n",
             (unsigned long)g_last_fb_get_ms, (unsigned)img_len, fb->format);

  // 4) 關燈
  setLEDBrightness(0);
  int64_t t_led_off = esp_timer_get_time();
  const double led_on_ms = (t_led_off - t_led_on) / 1000.0;
  DBG_PRINTF("[CAP][LED] off led_on_duration=%.1f ms\n", led_on_ms);
  DM_PRINTF("LED %.0fms\n", led_on_ms);
  releaseCaptureAutofocus();

#if COPY_FRAME_TO_PSRAM_BEFORE_SD
  // 5) 先複製 JPEG，讓 camera frame buffer 可以立即歸還並讓 sensor 進 standby。
  const int64_t t_copy_start = esp_timer_get_time();
  uint8_t *image_copy = (uint8_t *)heap_caps_malloc(img_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!image_copy) {
    image_copy = (uint8_t *)heap_caps_malloc(img_len, MALLOC_CAP_8BIT);
  }
  const uint8_t *write_buf = fb->buf;
  bool standby_already_entered = false;
  if (image_copy) {
    memcpy(image_copy, fb->buf, img_len);
    const int64_t t_copy_end = esp_timer_get_time();
    DBG_PRINTF("[CAP] copied JPEG to PSRAM in %.1f ms len=%u\n",
               (t_copy_end - t_copy_start) / 1000.0, (unsigned)img_len);
    write_buf = image_copy;
    cameraFbReturnLogged(fb, "final");
    fb = nullptr;
    setSensorStandby(true);
    standby_already_entered = true;
  } else {
    LOG.printf("[CAP] image copy alloc failed len=%u, fallback: standby after SD write\n", (unsigned)img_len);
  }
#else
  const uint8_t *write_buf = fb->buf;
  bool standby_already_entered = false;
#endif

  // 6) 寫入（原子）+ 驗證；失敗降速重掛一次
  double write_ms = 0.0, write_mbps = 0.0;
  bool ok = atomicWriteAndVerify(filePath, write_buf, img_len, &write_ms, &write_mbps);
  if (!ok) {
    LOG.println("[SD] write+verify failed, remount slower and retry once...");
    if (remountSdCard()) {
      ok = atomicWriteAndVerify(filePath, write_buf, img_len, &write_ms, &write_mbps);
    }
    if (!ok) {
      g_sd_ready = false;
    }
  }

  // 7) 釋放影像緩衝或複製緩衝
  if (fb) {
    cameraFbReturnLogged(fb, "final");
  }
#if COPY_FRAME_TO_PSRAM_BEFORE_SD
  if (image_copy) {
    heap_caps_free(image_copy);
  }
#endif
  int64_t t_after_fb_return = esp_timer_get_time();

  // 8) 時間資訊
  (void)t_after_fb_return;
  double led_to_fb_ms    = (t_got_fb - t_led_on) / 1000.0;
  double fb_to_ledoff_ms = (t_led_off - t_got_fb) / 1000.0;

  LOG.printf("Timings  LED->FB: %.1f ms, FB->LEDoff: %.1f ms, FB->SD(write): %.1f ms\n",
                led_to_fb_ms, fb_to_ledoff_ms, write_ms);
  LOG.printf("Size: %.2f MB, SD: %.2f MB/s  (SPI %.0f MHz)\n",
                img_len / (1024.0 * 1024.0), write_mbps, g_sd_hz / 1e6);

  if (ok) LOG.printf("[OK] Saved: %s\n", filePath);
  else    LOG.printf("[FAIL] Save failed: %s\n", filePath);
  DM_PRINTLN(ok ? "SAVE OK" : "SAVE FAIL");
  const float temp_c = readCpuTempC();
  if (!isnan(temp_c)) {
    DM_PRINTF("TMP %.1fC\n", temp_c);
  }
  if (!standby_already_entered) {
    setSensorStandby(true);
  }
  logRuntimeState(ok ? "after_capture_saved" : "after_capture_save_failed");
  const double total_elapsed_ms = (esp_timer_get_time() - t_photo_start) / 1000.0;
  DBG_PRINTF("[CAP] total_elapsed=%.1f ms\n", total_elapsed_ms);
  DM_PRINTF("TOT %.0fms\n", total_elapsed_ms);

  delay(10);
}

void setup() {
  LOG.begin(DEBUG_BAUD);
  delay(100);
  LOG.printf("[LOG] USB CDC + Serial1 enabled at %u baud (Serial1 RX=D7/GPIO44, TX=D6/GPIO43)\n", DEBUG_BAUD);
  DM_PRINTLN("BOOT");
  DM_PRINTF("UART %u\n", DEBUG_BAUD);

  SPI.begin();

  // LED：用 LEDC（避免與 camera LEDC 衝突）
  pinMode(LED_PIN, OUTPUT);
  led_init();
  statusBlink(1);

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
  config.frame_size   = CAPTURE_FRAME_SIZE;   // 需要更穩就先降 UXGA/SXGA
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count     = 1;                    // ⭐ 單緩衝

  if (esp_camera_init(&config) != ESP_OK) {
    LOG.println("[CAM] init failed");
    fatalBlinkLoop(2);
  }

  // 套用曝光/增益模式
  sensor_t *s = esp_camera_sensor_get();
  applyExposureMode(s);
  initAutofocus(s);
  warmupAndLockWhiteBalance(s);

  // 掛載 SD（多時脈嘗試）
  if (!sd_init_fast()) {
    LOG.println("[SD] mount failed (all speeds tried).");
    fatalBlinkLoop(3);
  }

  recoverImageCounter();

  LOG.println("Continuous mode: capture (no sleep).");
  logRuntimeState("setup_done");
}

void loop() {
  static const uint32_t period_ms = TARGET_PERIOD_S * 1000UL;

  // 用「絕對時間」排程：第一張立即拍，下一張對齊到 first_start + period。
  static uint32_t next_tick = 0;
  if (next_tick == 0) next_tick = millis();

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
      LOG.printf("[SCHED] behind %ld ms, skip %u slot(s) to realign.\n",
                    (long)behind, (unsigned)skip);
    } else {
      // 只是小落後（< 1 週期），不 skip，只記錄
      LOG.printf("[SCHED] minor slip %ld ms (no skip)\n", (long)behind);
    }
  }
}

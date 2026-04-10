#include "ov5640_af_support.h"
#include <ESP32_OV5640_AF.h>

namespace {

constexpr uint16_t OV5640_AF_CMD_MAIN = 0x3022;
constexpr uint16_t OV5640_AF_CMD_ACK = 0x3023;
constexpr uint16_t OV5640_AF_CMD_PARA0 = 0x3024;
constexpr uint16_t OV5640_AF_CMD_PARA1 = 0x3025;
constexpr uint16_t OV5640_AF_CMD_PARA2 = 0x3026;
constexpr uint16_t OV5640_AF_CMD_PARA3 = 0x3027;
constexpr uint16_t OV5640_AF_CMD_PARA4 = 0x3028;
constexpr uint16_t OV5640_AF_CMD_FW_STATUS = 0x3029;
constexpr int OV5640_AF_FW_NOT_READY = -2;
constexpr int OV5640_AF_UNSUPPORTED = -3;
constexpr uint8_t OV5640_AF_CMD_RELEASE = 0x08;
constexpr uint8_t OV5640_AF_CMD_CONTINUOUS = 0x04;
constexpr uint8_t OV5640_AF_CMD_SINGLE = 0x03;
constexpr uint8_t OV5640_AF_CMD_SET_ZONE = 0x90;
constexpr uint8_t OV5640_AF_CMD_SET_ZONE_WEIGHT = 0x98;
constexpr uint8_t OV5640_AF_CMD_ZONE_MODE = 0x12;
constexpr uint8_t OV5640_AF_STATUS_IDLE = 0x70;
constexpr uint8_t OV5640_AF_STATUS_FOCUSED = 0x10;

OV5640 ov5640;

sensor_t *g_sensor = nullptr;
bool g_af_ready = false;
int g_last_status = OV5640_AF_UNSUPPORTED;
int g_init_result = OV5640_AF_UNSUPPORTED;

bool write_af_cmd(uint8_t cmd) {
  if (!g_sensor || !g_af_ready) {
    return false;
  }
  return g_sensor->set_reg(g_sensor, OV5640_AF_CMD_MAIN, 0xFF, cmd) == 0;
}

int read_af_reg(uint16_t reg) {
  if (!g_sensor || !g_af_ready) {
    return -1;
  }
  return g_sensor->get_reg(g_sensor, reg, 0xFF);
}

bool wait_for_ack(uint8_t expected, uint16_t retries = 1000, uint16_t delay_ms = 5) {
  for (uint16_t i = 0; i < retries; ++i) {
    if (read_af_reg(OV5640_AF_CMD_ACK) == expected) {
      return true;
    }
    delay(delay_ms);
  }
  return false;
}

bool is_valid_af_runtime_status(int status) {
  return status == OV5640_AF_STATUS_IDLE || status == OV5640_AF_STATUS_FOCUSED || (status >= 0x00 && status <= 0x0F) || (status >= 0x80 && status <= 0x8F);
}

bool write_af_para(uint16_t reg, uint8_t value) {
  if (!g_sensor || !g_af_ready) {
    return false;
  }
  return g_sensor->set_reg(g_sensor, reg, 0xFF, value) == 0;
}

}  // namespace

bool ov5640_af_supported(sensor_t *sensor) {
  return sensor != nullptr && sensor->id.PID == OV5640_PID;
}

bool ov5640_af_begin(sensor_t *sensor) {
  g_sensor = sensor;
  g_af_ready = false;
  g_last_status = OV5640_AF_UNSUPPORTED;
  g_init_result = OV5640_AF_UNSUPPORTED;

  if (!ov5640_af_supported(sensor)) {
    return false;
  }

  ov5640.start(sensor);
  g_init_result = ov5640.focusInit();
  if (g_init_result != 0) {
    g_last_status = OV5640_AF_FW_NOT_READY;
    return false;
  }
  g_af_ready = true;
  g_last_status = ov5640.getFWStatus();
  return true;
}

bool ov5640_af_trigger_single() {
  return write_af_cmd(OV5640_AF_CMD_SINGLE);
}

bool ov5640_af_focus_zone(uint8_t x, uint8_t y, uint8_t w, uint8_t h) {
  if (!g_sensor || !g_af_ready) {
    return false;
  }

  const uint8_t max_x = 79;
  const uint8_t max_y = 59;
  uint8_t start_x = x > max_x ? max_x : x;
  uint8_t start_y = y > max_y ? max_y : y;
  uint8_t width = w == 0 ? 8 : w;
  uint8_t height = h == 0 ? 8 : h;
  uint8_t end_x = start_x + width;
  uint8_t end_y = start_y + height;
  if (end_x > max_x) {
    end_x = max_x;
  }
  if (end_y > max_y) {
    end_y = max_y;
  }

  if (!write_af_para(OV5640_AF_CMD_PARA0, start_x) || !write_af_para(OV5640_AF_CMD_PARA1, start_y) || !write_af_para(OV5640_AF_CMD_PARA2, end_x)
      || !write_af_para(OV5640_AF_CMD_PARA3, end_y)) {
    return false;
  }
  if (!write_af_cmd(OV5640_AF_CMD_SET_ZONE) || !wait_for_ack(0x00, 200, 5)) {
    return false;
  }
  if (!write_af_para(OV5640_AF_CMD_PARA0, 1) || !write_af_para(OV5640_AF_CMD_PARA1, 0) || !write_af_para(OV5640_AF_CMD_PARA2, 0)
      || !write_af_para(OV5640_AF_CMD_PARA3, 0) || !write_af_para(OV5640_AF_CMD_PARA4, 0)) {
    return false;
  }
  if (!write_af_cmd(OV5640_AF_CMD_SET_ZONE_WEIGHT) || !wait_for_ack(0x00, 200, 5)) {
    return false;
  }
  if (!write_af_cmd(OV5640_AF_CMD_ZONE_MODE) || !wait_for_ack(0x00, 200, 5)) {
    return false;
  }
  delay(20);
  return ov5640_af_trigger_single();
}

bool ov5640_af_enable_continuous() {
  if (!g_sensor || !g_af_ready) {
    return false;
  }
  int status = read_af_reg(OV5640_AF_CMD_FW_STATUS);
  if (is_valid_af_runtime_status(status)) {
    g_last_status = status;
  }

  if (ov5640.autoFocusMode() == 0) {
    g_last_status = ov5640.getFWStatus();
    return true;
  }

  // The library often times out when CAF is already active. Fall back to direct commands.
  if (!write_af_cmd(0x01)) {
    return false;
  }
  if (!write_af_cmd(OV5640_AF_CMD_RELEASE)) {
    return is_valid_af_runtime_status(read_af_reg(OV5640_AF_CMD_FW_STATUS));
  }
  if (!wait_for_ack(0x00, 200, 5)) {
    return is_valid_af_runtime_status(read_af_reg(OV5640_AF_CMD_FW_STATUS));
  }
  if (g_sensor->set_reg(g_sensor, OV5640_AF_CMD_ACK, 0xFF, 0x01) != 0) {
    return is_valid_af_runtime_status(read_af_reg(OV5640_AF_CMD_FW_STATUS));
  }
  if (!write_af_cmd(OV5640_AF_CMD_CONTINUOUS)) {
    return is_valid_af_runtime_status(read_af_reg(OV5640_AF_CMD_FW_STATUS));
  }
  if (!wait_for_ack(0x00, 200, 5)) {
    return is_valid_af_runtime_status(read_af_reg(OV5640_AF_CMD_FW_STATUS));
  }
  g_last_status = ov5640.getFWStatus();
  return is_valid_af_runtime_status(g_last_status);
}

bool ov5640_af_stop() {
  return write_af_cmd(0x06);
}

bool ov5640_af_release() {
  return write_af_cmd(0x08);
}

int ov5640_af_status() {
  if (!g_sensor || !g_af_ready) {
    return g_last_status;
  }
  g_last_status = ov5640.getFWStatus();
  return g_last_status;
}

int ov5640_af_init_result() {
  return g_init_result;
}

bool ov5640_af_library_available() {
  return true;
}

bool ov5640_af_ready() {
  return g_af_ready;
}

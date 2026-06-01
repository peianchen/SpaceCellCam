#pragma once

#include "esp_camera.h"

bool ov5640_af_supported(sensor_t *sensor);
bool ov5640_af_begin(sensor_t *sensor);
bool ov5640_af_trigger_single();
bool ov5640_af_set_focus_zone(uint8_t x, uint8_t y, uint8_t w, uint8_t h);
bool ov5640_af_focus_zone(uint8_t x, uint8_t y, uint8_t w, uint8_t h);
bool ov5640_af_enable_continuous();
bool ov5640_af_stop();
bool ov5640_af_release();
int ov5640_af_status();
int ov5640_af_init_result();
bool ov5640_af_library_available();
bool ov5640_af_ready();

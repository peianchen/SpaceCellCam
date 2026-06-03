# SpaceCellCam

Firmware for Seeed Studio XIAO ESP32S3 + OV5640 AF camera. It captures LED-assisted QSXGA JPEG images to microSD every 10 seconds.

Open `project_overview.html` for an interactive overview.

## Current Mode

Current firmware uses a focus-lock reuse mode with sensor standby:

```cpp
#define AF_LOCK_MODE      3
#define AF_KEEP_LOCK_AFTER_CAPTURE 1
#define AF_REFOCUS_EVERY_N 0
#define ENABLE_SENSOR_STANDBY_AFTER_CAPTURE 1
```

Behavior:

```text
First capture: run center AF and stop/lock focus
Later captures: reuse locked focus, do not run AF again
Every capture: copy JPEG to PSRAM, fb_return, enter OV5640 standby, then write SD
```

Important interpretation: if you hear VCM movement on later captures, that does not necessarily mean the firmware is re-focusing. With this mode, logs should show `reuse locked focus`; standby/wake may still cause the VCM or AF driver to move back to the previous focus position.

## Current Settings

```cpp
#define TARGET_PERIOD_S   10
#define XCLK_FREQ_HZ      20000000
#define CAPTURE_FRAME_SIZE FRAMESIZE_QSXGA
#define CAPTURE_RESTORE_SETTLE_MS 50
#define CAPTURE_DISCARD_FRAMES 1
#define LED_PREHEAT_MS    300
#define LED_BRIGHTNESS    80
```

AF region is a smaller center area to avoid LED ring/background influence:

```cpp
#define AF_CENTER_X       34
#define AF_CENTER_Y       24
#define AF_CENTER_W       12
#define AF_CENTER_H       10
```

SD card speeds:

```cpp
40 / 32 / 25 / 20 / 16 / 12 MHz
```

Files are saved as:

```text
/bXXX/image_<n>.jpg
```

`BUCKET_SIZE = 500`, and image numbering is recovered from SD at boot to avoid overwriting old images.

## Capture Flow

1. Wake OV5640 from standby if needed.
2. Turn LED on at brightness 80.
3. Wait 300 ms for LED/exposure stability.
4. If focus is not locked yet, run tap-style center AF using QVGA pump frames.
5. If focus is already locked, skip AF and reuse the previous focus state.
6. Discard 1 frame.
7. Capture final QSXGA JPEG.
8. Turn LED off.
9. Keep focus locked; do not release VCM in this mode.
10. Copy JPEG to PSRAM.
11. Return camera frame buffer.
12. Enter OV5640 standby with register `0x3008 = 0x42`.
13. Write copied JPEG to SD using temp-file write, rename, and size verification.

## What To Look For In Logs

First capture should contain:

```text
[AF][CAP] start tap-style center focus
[AF] per-capture focus confirmed
[AF] keep focus locked after capture
```

Later captures should contain:

```text
[AF][CAP] reuse locked focus from img=...
[AF] keep focus locked after capture
```

If you see this, the firmware is not running a new AF search on that capture:

```text
[AF][CAP] reuse locked focus
```

If you see this, the firmware is running a new AF search:

```text
[AF][CAP] start tap-style center focus
[AF] per-capture focus confirmed
```

Standby should still happen after each capture:

```text
[CAM][STANDBY] enter ok reg=0x3008 value=0x42
[RUNTIME] standby_entered ... standby=1
```

Wake should happen before the next capture:

```text
[CAM][STANDBY] wake ok reg=0x3008 value=0x02
```

## Verification: Reuse Lock vs Re-Focus

To verify that later captures are not re-focusing:

1. Put the target at distance A.
2. Let the first image focus.
3. Capture the second image without moving; it should be clear.
4. Move the target to distance B before the next capture.
5. Check the log and photo.

Expected result if focus is truly reused:

```text
Log: [AF][CAP] reuse locked focus
Photo at distance B: blurrier than distance A
```

If the target at distance B becomes clear while the log still says `reuse locked focus`, that is probably not firmware re-focus. It may be VCM/AF driver state recovery during standby/wake, or the scene may have enough depth of field that both distances look acceptable.

For a stronger test, use a larger distance change or a slanted ruler/board so the focus plane is visible.

## Debug Mate UART

Wiring:

```text
XIAO D6 / GPIO43 / TX  ->  Debug Mate RX
XIAO D7 / GPIO44 / RX  ->  Debug Mate TX
XIAO GND              ->  Debug Mate GND
Baud                  ->  115200
Level                 ->  3.3V UART
```

If the XIAO is externally powered, connect only `TX`, `RX`, and `GND`.

Debug Mate receives short status lines:

```text
BOOT
UART 115200
SD 40M OK
CAP image_5936.jpg
INT 10.0s
WAKE
LED 665ms
STBY ON
SAVE OK
TMP 36.5C
TOT 1673ms
```

USB CDC still receives the full log.

## Timing Expectations

With focus reuse:

```text
LED on duration: about 0.65 s
Total capture work: about 1.6 - 1.8 s
INT: about 10.0 s
```

With every-shot AF:

```text
LED on duration: about 2.1 s
Total capture work: about 3.1 s
```

SD write currently happens after LED off and after sensor standby, so SD time affects total work but not illumination time.

## Standby And Heat

The current mode keeps the firmware focus lock but still puts OV5640 into standby every shot. This is the current compromise:

```text
Less LED time than every-shot AF
Sensor standby still active
No AF search on later captures
VCM may still make a movement sound during wake/recovery
```

CPU temperature in logs is ESP32-S3 internal temperature, not OV5640 or VCM temperature.

## Power Notes

Unstable USB power can cause frame timeouts, corrupted images, long discard times, or `esp_camera_fb_get()` failure. For capture tests, use stable external power and Debug Mate UART instead of relying on PC USB power.

## Build

```powershell
C:\Users\chen8\.platformio\penv\Scripts\pio.exe run
```

Target:

```ini
[env:seeed-xiao-esp32-s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 115200
```

## Troubleshooting

AF failure:

```text
[AF] capture skipped because focus was not confirmed
```

Frame failure:

```text
[CAM] esp_camera_fb_get() failed
```

SD failure:

```text
[SD] write short
[SD] sync failed
[SD] verify size mismatch
```

Memory check:

```text
[RUNTIME] standby_entered ... free_psram=718xxxx
[RUNTIME] after_capture_saved ... free_psram=7402739
```

The temporary PSRAM drop is expected while the JPEG copy is waiting for SD write. It should recover after save completes.

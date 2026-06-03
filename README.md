# SpaceCellCam

Firmware for a Seeed Studio XIAO ESP32S3 + OV5640 AF camera that captures LED-assisted QSXGA JPEG images to microSD at a fixed interval.

Open `project_overview.html` for an interactive project overview.

## Current Firmware Snapshot

- Board: Seeed Studio XIAO ESP32S3.
- Camera: OV5640 with AF / VCM support.
- Capture size: `FRAMESIZE_QSXGA`.
- Capture period: `TARGET_PERIOD_S = 10` seconds.
- XCLK: `XCLK_FREQ_HZ = 20000000`.
- LED: `LED_BRIGHTNESS = 60`, `LED_PREHEAT_MS = 200`.
- SD card: SPI mode, tries `40 / 32 / 25 / 20 / 16 / 12 MHz`.
- File layout: `/bXXX/image_<n>.jpg`, `BUCKET_SIZE = 500`.
- Image counter is recovered from SD at boot, so reboot continues from the next image number.
- USB CDC keeps the full debug log.
- `Serial1` on `D6` / `D7` sends short Debug Mate status lines.

## Capture Flow

1. Wake OV5640 from standby if needed.
2. Turn on LED at `LED_BRIGHTNESS = 60`.
3. Wait `LED_PREHEAT_MS = 200`.
4. Run per-capture center autofocus.
5. Temporarily switch to QVGA for fast AF pump.
6. Restore QSXGA and wait `CAPTURE_RESTORE_SETTLE_MS = 50`.
7. Discard `CAPTURE_DISCARD_FRAMES = 1`.
8. Capture final QSXGA JPEG.
9. Turn off LED and release AF.
10. Copy JPEG to PSRAM.
11. Return camera frame buffer and enter OV5640 standby.
12. Write JPEG to SD with temp-file write, rename, and size verification.

This keeps LED exposure short and lets the camera sensor enter standby before the slower SD write completes.

## Key Settings

```cpp
#define TARGET_PERIOD_S   10
#define CAPTURE_FRAME_SIZE FRAMESIZE_QSXGA
#define CAPTURE_RESTORE_SETTLE_MS 50
#define CAPTURE_DISCARD_FRAMES 1
#define LED_PREHEAT_MS    200
#define LED_BRIGHTNESS    60
```

```cpp
#define ENABLE_AUTOFOCUS  1
#define AF_LOCK_MODE      3
#define AF_CAPTURE_SETTLE_MS 2500
#define AF_PUMP_FRAME_SIZE FRAMESIZE_QVGA
#define AF_PUMP_FRAME_COUNT 5
#define AF_KEEP_LOCK_AFTER_CAPTURE 0
#define AF_STABLE_COUNT   1
```

```cpp
#define ENABLE_SENSOR_STANDBY_AFTER_CAPTURE 1
#define COPY_FRAME_TO_PSRAM_BEFORE_SD 1
#define SENSOR_STANDBY_SETTLE_MS 80
#define SENSOR_WAKE_SETTLE_MS 120
```

## Autofocus

The current mode is per-capture center AF:

```cpp
#define AF_CENTER_X       30
#define AF_CENTER_Y       22
#define AF_CENTER_W       20
#define AF_CENTER_H       16
```

`AF_LOCK_MODE = 3` means every image triggers AF again. `AF_KEEP_LOCK_AFTER_CAPTURE = 0` releases AF after each shot so the VCM is not held energized continuously. This is slower than keeping focus locked, but matches the current power and heating goal.

AF wait-loop spam is disabled:

```cpp
#define AF_DEBUG_WAIT_LOG 0
#define AF_DEBUG_EVERY_POLL 0
```

## Standby Behavior

OV5640 standby is controlled by register `0x3008`:

```cpp
s->set_reg(s, 0x3008, 0xFF, 0x42); // standby
s->set_reg(s, 0x3008, 0xFF, 0x02); // wake
```

Normal logs:

```text
[CAM][STANDBY] enter ok reg=0x3008 value=0x42
[RUNTIME] standby_entered ... standby=1
[CAM][STANDBY] wake ok reg=0x3008 value=0x02
```

Because `COPY_FRAME_TO_PSRAM_BEFORE_SD = 1`, the camera frame buffer is returned and the sensor enters standby before SD writing. PSRAM should temporarily drop while holding the JPEG copy and return after save completes.

## Debug Mate UART

Wiring:

```text
XIAO D6 / GPIO43 / TX  ->  Debug Mate RX
XIAO D7 / GPIO44 / RX  ->  Debug Mate TX
XIAO GND              ->  Debug Mate GND
Baud                  ->  115200
Level                 ->  3.3V UART
```

If the XIAO is already powered from a stable external supply, connect only `TX`, `RX`, and `GND`. Do not feed 5V into UART pins.

Debug Mate receives short lines only:

```text
BOOT
UART 115200
SD 40M OK
CAP image_63.jpg
INT 10.0s
WAKE
AF 1196ms
LED 2089ms
STBY ON
SAVE OK
TMP 33.5C
TOT 2853ms
```

USB CDC `Serial` still receives the full detailed log.

## Normal Full Log Pattern

```text
[CAP][INTERVAL] capture=... img=... interval=10000 ms target=10000 ms
[RUNTIME] before_capture ... standby=1 ...
[CAM][STANDBY] wake ok reg=0x3008 value=0x02
[CAP][LED] on brightness=60 preheat=200 ms
[AF][CAP] start tap-style center focus
[AF][CAP] frame size switched for fast AF pump
[AF] per-capture focus confirmed, elapsed=...
[AF][CAP] frame size restored for QSXGA capture
[CAP] discarded 1 frames in ...
[CAP] fb_get=447 ms len=... bytes format=4
[CAP][LED] off led_on_duration=...
[AF] per-capture released
[CAP] copied JPEG to PSRAM in ...
[CAM][STANDBY] enter ok reg=0x3008 value=0x42
[OK] Saved: /b000/image_...
[RUNTIME] after_capture_saved ... standby=1 ...
```

To confirm the camera is idle between captures, check that `fb_get` does not increase between `after_capture_saved` and the next `before_capture`, and that `mjpeg_stream=0`.

## Timing Expectations

Recent normal values:

- `INT`: about `10.0s`.
- AF: about `1.1~1.3s`.
- LED on duration: about `2.0~2.1s`.
- JPEG copy to PSRAM: about `12ms`.
- SD write at 40 MHz: about `0.45s` for a 0.25 MB JPEG.
- Total capture work: about `2.8s`.

First capture after boot may have different interval display. From the second capture onward, `INT` should be close to 10 seconds unless a failure or blocking operation delays the loop.

## Power Notes

Unstable USB power or noisy PC USB connections can cause camera frame timeouts, corrupted-looking JPEGs, slow discard frames, or `esp_camera_fb_get()` failure.

Known symptoms:

```text
[CAP] discarded 1 frames in 8000+ ms
[CAM] esp_camera_fb_get() failed
```

For capture testing, use a stable external power source. If logs are needed without PC USB power, use Debug Mate UART on `D6` / `D7`.

## Speed Tuning

Current conservative-fast settings:

```cpp
#define CAPTURE_RESTORE_SETTLE_MS 50
#define CAPTURE_DISCARD_FRAMES 1
#define AF_PUMP_FRAME_COUNT 5
#define AF_STABLE_COUNT 1
```

Possible further tests:

1. `CAPTURE_RESTORE_SETTLE_MS 50 -> 20`
2. `SENSOR_WAKE_SETTLE_MS 120 -> 80`
3. `CAPTURE_DISCARD_FRAMES 1 -> 0`

Do not change many timing values at once. Capture a short run, compare images, and check `INT`, `AF`, `LED`, `STBY`, and `TOT` on Debug Mate.

## Build

Use PlatformIO:

```powershell
C:\Users\chen8\.platformio\penv\Scripts\pio.exe run
```

Project target:

```ini
[env:seeed-xiao-esp32-s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 115200
```

## Troubleshooting

AF issue:

```text
[AF] capture skipped because focus was not confirmed
```

Camera frame issue:

```text
[CAM] esp_camera_fb_get() failed
```

SD issue:

```text
[SD] write short
[SD] sync failed
[SD] verify size mismatch
[SD] write+verify failed
```

Memory check:

```text
[RUNTIME] standby_entered ... free_psram=713xxxx
[RUNTIME] after_capture_saved ... free_psram=7402739
```

The lower value during `standby_entered` is expected while the copied JPEG is still held for SD writing. It should return after save completes.

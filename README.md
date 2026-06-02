# SpaceCellCam

XIAO ESP32S3 Sense / OV5640 AF camera firmware for periodic LED-assisted image capture to microSD.

The current firmware is a continuous capture build: it does not sleep between shots. Each capture turns on the LED, runs center autofocus, captures a QSXGA JPEG, writes it to SD with verification, then releases autofocus power before the next shot.

For a visual interactive overview, open `project_overview.html` in a browser.

## Current Behavior

- Board: Seeed Studio XIAO ESP32S3.
- Camera: OV5640 with autofocus / VCM support.
- Capture size: `FRAMESIZE_QSXGA`.
- Capture period: `TARGET_PERIOD_S = 10`.
- XCLK: `XCLK_FREQ_HZ = 20000000`.
- SD card: SPI mode, tries 25 / 20 / 16 / 12 MHz.
- File layout: `/bXXX/image_<n>.jpg`, with `BUCKET_SIZE = 500` images per folder.
- Image counter is recovered from the SD card at boot, so reboot should continue at the next number instead of overwriting old images.
- Debug output is sent to both USB CDC `Serial` and UART `Serial1`.

## Capture Flow

1. Turn on capture LED at `LED_BRIGHTNESS = 40`.
2. Wait `LED_PREHEAT_MS = 200`.
3. Run per-capture autofocus at the center zone.
4. Switch to QVGA for fast AF pump, then restore QSXGA.
5. Wait `CAPTURE_RESTORE_SETTLE_MS = 120` after restoring QSXGA.
6. Discard `CAPTURE_DISCARD_FRAMES = 2` frames.
7. Capture the final JPEG frame.
8. Turn off LED.
9. Release autofocus with `ov5640_af_release()`.
10. Write the image with atomic temp-file write, rename, and size verification.

## Autofocus Settings

The current AF mode is per-capture center focus:

```cpp
#define ENABLE_AUTOFOCUS  1
#define AF_LOCK_MODE      3
#define AF_CENTER_X       30
#define AF_CENTER_Y       22
#define AF_CENTER_W       20
#define AF_CENTER_H       16
```

The AF zone is a fixed center area on the OV5640 AF grid. `AF_LOCK_MODE = 3` means every photo triggers AF again.

The current build releases AF after every shot:

```cpp
#define AF_KEEP_LOCK_AFTER_CAPTURE 0
```

This avoids keeping the VCM energized continuously. It is intentionally slower than keeping focus locked, but better matches the current hardware/power goal.

Fast AF pump is enabled:

```cpp
#define AF_SWITCH_FRAME_SIZE_FOR_PUMP 1
#define AF_PUMP_FRAME_SIZE FRAMESIZE_QVGA
#define AF_PUMP_FRAME_COUNT 6
#define AF_CAPTURE_SETTLE_MS 2500
```

During AF the camera temporarily switches to QVGA to feed frames faster to the AF firmware, then switches back to QSXGA before the final capture.

## Debug Output

Debug output goes to both:

- USB CDC serial
- `Serial1` on XIAO pins `D6` / `D7`

UART wiring for XIAO Debug Mate or a USB-UART adapter:

```text
XIAO D6 / GPIO43 / TX  ->  Debug adapter RX
XIAO D7 / GPIO44 / RX  ->  Debug adapter TX
XIAO GND              ->  Debug adapter GND
Baud rate             ->  115200
```

If the XIAO is already powered from a separate stable supply, do not also power it from the debug adapter. Connect only TX, RX, and GND.

The detailed AF wait loop log is currently disabled to avoid slowing capture:

```cpp
#define AF_DEBUG_WAIT_LOG 0
#define AF_DEBUG_EVERY_POLL 0
```

Important summary logs are still printed, for example:

```text
[AF] per-capture focus confirmed
[CAP] discarded 2 frames in ...
[CAP] fb_get=...
[OK] Saved: ...
```

## Power Notes

Unstable USB power or noisy PC USB connections can cause camera frame timeouts, corrupted-looking JPEGs, or very slow `esp_camera_fb_get()` behavior.

Known symptom:

```text
[CAP] discarded 2 frames in 8000+ ms
[CAM] esp_camera_fb_get() failed
```

If this happens, test with a stable external supply and avoid powering the board from the PC USB port. Use UART logging through `D6` / `D7` if logs are needed while externally powered.

## Speed Tuning

The current settings are conservative for stable QSXGA capture after AF:

```cpp
#define CAPTURE_RESTORE_SETTLE_MS 120
#define CAPTURE_DISCARD_FRAMES 2
#define AF_PUMP_FRAME_COUNT 6
#define AF_STABLE_COUNT 2
```

Possible speed-up tests, from lower risk to higher risk:

1. `CAPTURE_RESTORE_SETTLE_MS 120 -> 50`
2. `CAPTURE_DISCARD_FRAMES 2 -> 1`
3. `AF_PUMP_FRAME_COUNT 6 -> 5`
4. `AF_STABLE_COUNT 2 -> 1`

Recommended first trial:

```cpp
#define CAPTURE_RESTORE_SETTLE_MS 50
#define CAPTURE_DISCARD_FRAMES 1
#define AF_PUMP_FRAME_COUNT 5
#define AF_STABLE_COUNT 2
```

Do not change all timing parameters at once when debugging. Change one or two values, capture a short run, then compare logs and images.

## Build

Use PlatformIO:

```powershell
C:\Users\chen8\.platformio\penv\Scripts\pio.exe run
```

The project target is configured in `platformio.ini`:

```ini
[env:seeed-xiao-esp32-s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
```

## Useful Log Checks

Normal capture should look roughly like:

```text
[CAP] begin img=...
[CAP][LED] on brightness=40 preheat=200 ms
[AF][CAP] start tap-style center focus
[AF][CAP] frame size switched for fast AF pump
[AF] per-capture focus confirmed
[AF][CAP] frame size restored for QSXGA capture
[CAP] discarded 2 frames in ...
[CAP] fb_get=...
[CAP][LED] off led_on_duration=...
[AF] per-capture released
[OK] Saved: ...
```

If AF succeeds but no image is saved, check for:

```text
[CAM] esp_camera_fb_get() failed
```

If SD is the problem, check for:

```text
[SD] write short
[SD] sync failed
[SD] verify size mismatch
[SD] write+verify failed
```

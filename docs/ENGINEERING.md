# SpaceCellCam: Engineering notes

[README](../README.md) · [中文](ENGINEERING.zh-TW.md) · [English](ENGINEERING.md)

ESP32-S3 and OV5640 AF firmware coordinates lighting, focus reuse, JPEG capture and microSD storage for observable experimental imaging.

## Purpose

Long-running biological observation needs an explicit capture sequence and recoverable images. This firmware schedules the camera, LED, memory and storage as one cycle, with logs that distinguish focus, capture, standby and file errors.

## Architecture and workflow

```text
WAKE + LIGHT -> FOCUS + JPEG -> PSRAM + STANDBY -> SD + VERIFY
```

### Wake and illuminate

Current configuration targets a 10-second period and 20 MHz XCLK. The sensor wakes and the LED settles for 300 ms before capture. These are configuration values.

### Focus and capture

The first capture uses center autofocus and locks it. Later captures reuse that lock, discard one transition frame and acquire a QSXGA JPEG.

### Release the frame buffer

The JPEG is copied into PSRAM before returning the camera buffer. The sensor can then enter standby before SD writing begins.

### Save and verify

A temporary file is written, renamed and checked for size. Each bucket holds 500 images; numbering is recovered at boot to avoid overwriting previous captures.


## Design decisions

### Reuse focus with a scene constraint

A fixed scene can avoid repeated autofocus searches. A changing target distance may instead produce blur, so images and logs must be interpreted together.

### Trade memory for decoupled work

A PSRAM copy requires temporary memory but lets SD writes happen after LED off and sensor standby.

### Preserve failure clues

Focus failure, frame timeout, short writes and size mismatches have distinct logs to localize the failing stage.


## Results and evidence

### Documented configuration

The public README and firmware describe the 10-second target, focus reuse, standby and temporary-file verification.

### How to interpret a run

Compare reuse-locked-focus, standby enter/wake and SD verification messages against the actual captured images.

### Limits of this review

No new capture experiment was performed for this page. Configuration alone cannot establish achieved timing, power, optical quality or long-run data completeness.

## Reproduction

1. Prepare a XIAO ESP32S3, OV5640 AF, microSD and stable power, then build with the repository's platformio.ini.
2. Record first and subsequent captures and distinguish the autofocus-search and focus-reuse log paths.
3. Change target distance after establishing a fixed scene, compare sharpness and logs, and inspect file sizes, numbering and restart behavior.

## Limits

VCM noise during standby/wake does not by itself mean a new autofocus search. ESP32-S3 internal temperature is not camera or VCM temperature. The project name does not establish space-environment or orbital qualification.

# M5Stack K210 Camera — GuideLens On-Device Navigation
# =====================================================

> **NOTE** : I had this camera on me and hence had the idea of using it. It works fine without the camera and just using a webcam as well.


This folder contains everything needed to flash the M5Stack UnitV K210
camera for **standalone GuideLens navigation** mode.

## Architecture

```
┌──────────────────────────────────┐
│     M5Stack K210 Camera          │
│  ┌────────────┐  ┌────────────┐  │
│  │ OV2640 cam │→ │ YOLO v2    │  │     USB/UART         ┌──────────────┐
│  │ 224×224    │  │ tiny 20cls │──│────────────────────── │ Host Computer│
│  └────────────┘  │ .kmodel    │  │  Detection JSON      │              │
│                  └────────────┘  │  + JPEG frames        │ camera_host/ │
│  On-device inference @ ~15 FPS   │                       │ bridge.py    │
│  KPU accelerator (0.8 TOPS)      │                       │ → port 8001  │
└──────────────────────────────────┘                       └──────┬───────┘
                                                                  │
                                                           WebSocket stream
                                                                  │
                                                           ┌──────▼───────┐
                                                           │ Frontend UI  │
                                                           │ localhost:   │
                                                           │ 5173 or 8001 │
                                                           └──────────────┘
```

## What runs WHERE

| Component | Where | What |
|-----------|-------|------|
| `camera/boot.py` | K210 flash | Auto-start detection on power-on |
| `camera/main.py` | K210 flash | YOLO inference + UART output |
| `camera/config.py` | K210 flash | Camera & detection settings |
| `camera_host/bridge.py` | Your laptop | Serial → WebSocket bridge + HTTP UI server on port 8001 |
| `models/` | K210 SD card or flash | .kmodel files |

## Hardware Requirements

- **M5Stack UnitV K210** AI Camera (the one you have)
- **USB-C cable** connecting camera to laptop
- **microSD card** (optional, for larger models — 8GB or 16GB FAT32)

## K210 Specs vs Requirements

| Resource | Available | Used |
|----------|-----------|------|
| SRAM | 8 MiB | ~2-3 MiB (model + frame buffer) |
| Flash | 16 MB | ~4 MB (firmware + model) |
| KPU | 0.8 TOPS | YOLO v2 tiny fits perfectly |
| Model size limit | 5.9 MiB | ~1.3 MiB (20-class YOLO v2 tiny) |

## Quick Start

### Step 1: Flash MaixPy Firmware

1. Download [MaixPy firmware](https://dl.sipeed.com/MAIX/MaixPy/release/master/) 
   (choose `maixpy_v0.6.2_xx_minimum_with_kmodel_v4_support.bin`)
2. Download [kflash_gui](https://github.com/sipeed/kflash_gui/releases)
3. Connect K210 via USB-C
4. Flash the firmware using kflash_gui:
   - Board: `M5StickV / UnitV`
   - Port: Your serial port (e.g., `/dev/tty.usbserial-xxxx`)
   - Baud: 1500000
   - Firmware: the .bin you downloaded

### Step 2: Get the YOLO Model

Download the pre-trained 20-class YOLO v2 tiny `.kmodel`:

```bash
# Option A: Download from Sipeed model zoo
cd m5stack_camera/models
curl -L -o yolo_20class.kmodel \
  "https://dl.sipeed.com/fileList/MAIX/MaixPy/model/mobilenet_yolo/yolo_20class.kmodel"

# Option B: If you have an SD card, copy the .kmodel to the SD root
cp yolo_20class.kmodel /Volumes/YOUR_SD_CARD/
```

The 20-class model detects the same COCO subset useful for navigation:
person, bicycle, car, motorbike, bus, truck, cat, dog, chair, sofa,
tv, laptop, bottle, dining table, bird, boat, aeroplane, train, horse, sheep.

### Step 3: Upload Code to K210

Using [mpfshell](https://github.com/wendlers/mpfshell) or [ampy](https://github.com/scientifictoolworks/ampy):

```bash
# Install ampy
pip install adafruit-ampy

# Find your serial port
ls /dev/tty.usbserial-*   # macOS
ls /dev/ttyUSB*            # Linux

# Upload files to K210 flash
export M5_PORT=/dev/tty.usbserial-XXXX

ampy -p $M5_PORT put m5stack_camera/camera/boot.py /flash/boot.py
ampy -p $M5_PORT put m5stack_camera/camera/main.py /flash/main.py
ampy -p $M5_PORT put m5stack_camera/camera/config.py /flash/config.py

# If model is on flash (not SD card):
ampy -p $M5_PORT put m5stack_camera/models/yolo_20class.kmodel /flash/yolo_20class.kmodel
```

### Step 4: Run the Host Bridge

On your laptop (receives camera detections + frames via USB serial):

```bash
cd m5stack_camera/camera_host

# Install dependencies
pip install pyserial websockets aiohttp pillow

# Run the bridge (auto-detects K210 serial port)
python bridge.py --port /dev/tty.usbserial-XXXX

# Or auto-detect:
python bridge.py --auto
```

This starts:
- **WebSocket server** on `ws://localhost:8001/ws` — streams detections + frames
- **HTTP server** on `http://localhost:8001` — serves a standalone camera UI

### Step 5: Open the Camera UI

Open `http://localhost:8001` in your browser. You'll see:
- Live camera feed from the K210
- Bounding box overlays for detected objects
- Hazard alerts (person/car/bicycle approaching)
- Direction indicators (left/center/right)

This UI is **completely standalone** — it does NOT touch the main WorldLens
backend on port 8000.

## Wearing It

1. Mount the K210 camera on your glasses (use tape, mount, or 3D-printed clip)
2. Connect USB-C from camera to laptop (keep laptop in backpack)
3. Power on — camera auto-starts detection
4. Open browser on laptop/phone → `http://localhost:8001`
5. The bridge receives detections over USB serial and displays them in the UI
6. You can also screen-record the UI for demo purposes

## File Structure

```
m5stack_camera/
├── README.md                  ← This file
├── camera/                    ← Code that runs ON the K210
│   ├── boot.py               ← MaixPy auto-start
│   ├── main.py               ← YOLO inference + UART output
│   └── config.py             ← Detection settings
├── camera_host/               ← Code that runs on your LAPTOP
│   ├── bridge.py             ← Serial → WebSocket bridge + HTTP UI server
│   ├── requirements.txt      ← Python dependencies
│   └── static/
│       └── index.html        ← Camera viewer UI (port 8001)
└── models/
    ├── README.md             ← Model download instructions
    └── .gitkeep              ← Models not committed (too large)
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Serial port not found | Run `ls /dev/tty.usb*` and use the correct port |
| Camera shows green/garbage | Re-flash MaixPy firmware with kflash_gui |
| Model not loading | Ensure `.kmodel` is in `/flash/` or SD card root |
| Low FPS (<10) | Reduce resolution in `config.py` to 160×120 |
| ampy timeout | Use `--baud 115200` and try again |
| SD card not detected | Must be ≤16GB, FAT32, Class 10 (Kingston/SanDisk recommended) |

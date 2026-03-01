# K210 YOLO Models
# =================
# Models are not committed to git (too large).
# Download the 20-class YOLO v2 tiny .kmodel for the K210:

## Download Links

### Option 1: Sipeed Model Zoo (recommended)
```bash
curl -L -o yolo_20class.kmodel \
  "https://dl.sipeed.com/fileList/MAIX/MaixPy/model/mobilenet_yolo/yolo_20class.kmodel"
```

### Option 2: MaixHub
Visit https://maixhub.com/model/zoo and search for "yolo 20class".

## Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLO v2 tiny |
| Input size | 224 × 224 |
| Classes | 20 (COCO subset) |
| File size | ~1.3 MiB |
| Format | .kmodel (Kendryte KPU) |
| Output shape | 7 × 7 × 125 |

## 20 Classes
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
dining table, dog, horse, motorbike, person, potted plant, sheep,
sofa, train, tv

## Navigation-relevant Classes
For GuideLens navigation, the most important detections are:
- **person** — pedestrians
- **bicycle** — cyclists
- **car** — vehicles
- **motorbike** — motorcycles
- **bus** — buses
- **dog** — animals on path
- **chair** — obstacles

## Where to Place
- **SD Card**: Copy to root of FAT32 microSD card → `/sd/yolo_20class.kmodel`
- **Flash**: Upload via ampy → `/flash/yolo_20class.kmodel`

SD card is recommended (faster to update, doesn't use limited flash space).

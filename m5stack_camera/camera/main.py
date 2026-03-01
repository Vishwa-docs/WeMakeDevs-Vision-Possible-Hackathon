"""
GuideLens K210 — Main Detection Loop
======================================
Runs YOLO v2 tiny object detection on the K210's KPU hardware accelerator.
Outputs detection results as JSON over USB serial (UART) for the host
bridge to consume. Optionally sends JPEG frame snapshots.

Protocol (USB Serial, line-based JSON):
  Detection:  {"t":"det","f":42,"objs":[{"c":"car","x":0.3,"y":0.5,"w":0.2,"h":0.3,"p":0.87,"d":"left","dist":"near"}]}
  Frame:      {"t":"jpg","f":42,"sz":1234}\n<raw JPEG bytes>
  Heartbeat:  {"t":"hb","f":42,"fps":14.2,"mem":123456}
  Error:      {"t":"err","msg":"..."}
"""

import gc
import time
import json
import sys
import os

import sensor
import image
import lcd
import KPU as kpu

# Import config (on flash alongside this file)
try:
    from config import *
except ImportError:
    # Fallback defaults if config.py is missing
    CAMERA_WIDTH = 224
    CAMERA_HEIGHT = 224
    MODEL_PATH = "/sd/yolo_20class.kmodel"
    MODEL_FLASH_PATH = "/flash/yolo_20class.kmodel"
    CONFIDENCE_THRESHOLD = 0.45
    NMS_THRESHOLD = 0.3
    CLASS_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "dining table", "dog", "horse", "motorbike", "person",
        "potted plant", "sheep", "sofa", "train", "tv"
    ]
    HAZARD_CLASSES = {
        "person", "bicycle", "car", "motorbike", "bus",
        "dog", "horse", "chair"
    }
    SEND_JPEG_FRAMES = True
    JPEG_QUALITY = 40
    JPEG_SEND_INTERVAL = 3
    LED_ENABLED = True
    LED_HAZARD_COLOR = (255, 0, 0)
    LED_DETECTION_COLOR = (0, 255, 0)
    LED_IDLE_COLOR = (0, 0, 50)
    DISTANCE_NEAR = 0.15
    DISTANCE_MEDIUM = 0.05
    DIRECTION_LEFT = 0.33
    DIRECTION_RIGHT = 0.66


# ─────────────────────────────────────────
# LED setup (WS2812 on the UnitV)
# ─────────────────────────────────────────
_led = None
try:
    if LED_ENABLED:
        from modules import ws2812
        _led = ws2812(8, 1)  # Pin 8, 1 LED
        _led.set_led(0, LED_IDLE_COLOR)
        _led.display()
except Exception:
    _led = None


def set_led(color):
    """Set the onboard RGB LED color."""
    if _led:
        try:
            _led.set_led(0, color)
            _led.display()
        except Exception:
            pass


# ─────────────────────────────────────────
# Camera sensor init
# ─────────────────────────────────────────
def init_camera():
    """Initialize the OV2640 camera sensor."""
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)  # 320x240, will be cropped to 224x224
    sensor.set_windowing((CAMERA_WIDTH, CAMERA_HEIGHT))
    sensor.set_vflip(False)
    sensor.set_hmirror(False)
    sensor.run(1)
    # Let sensor stabilize
    for _ in range(10):
        sensor.snapshot()
        time.sleep_ms(50)
    print("[GuideLens] Camera initialized: {}x{}".format(CAMERA_WIDTH, CAMERA_HEIGHT))


# ─────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────
def load_model():
    """Load the YOLO v2 tiny .kmodel onto the KPU."""
    # Try SD card first, then flash
    model_path = MODEL_PATH
    try:
        os.stat(model_path)
    except OSError:
        model_path = MODEL_FLASH_PATH
        try:
            os.stat(model_path)
        except OSError:
            send_error("Model not found at {} or {}".format(MODEL_PATH, MODEL_FLASH_PATH))
            return None

    print("[GuideLens] Loading model: {}".format(model_path))
    gc.collect()

    try:
        task = kpu.load(model_path)
        kpu.set_outputs_shape(task, 0, 7, 7, 125)  # YOLO v2 tiny output shape
        # Anchor boxes for 20-class YOLO v2 tiny
        anchor = (
            1.08, 1.19,
            3.42, 4.41,
            6.63, 11.38,
            9.42, 5.11,
            16.62, 10.52
        )
        kpu.init_yolo2(task, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, 5, anchor)
        print("[GuideLens] Model loaded successfully")
        return task
    except Exception as e:
        send_error("Model load failed: {}".format(e))
        return None


# ─────────────────────────────────────────
# Serial output helpers
# ─────────────────────────────────────────
def send_json(data):
    """Send a JSON line over USB serial."""
    try:
        line = json.dumps(data)
        sys.stdout.write(line + "\n")
    except Exception:
        pass


def send_error(msg):
    """Send an error message over serial."""
    send_json({"t": "err", "msg": str(msg)})


def send_jpeg(img, frame_num):
    """Send a JPEG-compressed frame over serial."""
    try:
        jpg_bytes = img.compress(quality=JPEG_QUALITY)
        header = json.dumps({"t": "jpg", "f": frame_num, "sz": len(jpg_bytes)})
        sys.stdout.write(header + "\n")
        sys.stdout.buffer.write(jpg_bytes)
    except Exception as e:
        send_error("JPEG send failed: {}".format(e))


# ─────────────────────────────────────────
# Detection analysis
# ─────────────────────────────────────────
def analyse_detection(det, frame_w, frame_h):
    """
    Analyse a single YOLO detection and return a structured dict.

    det format from KPU: (x, y, w, h, class_id, confidence)
    Note: x, y, w, h are in pixels.
    """
    x, y, w, h, cls_id, conf = det
    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"

    # Normalize to 0-1
    cx = (x + w / 2) / frame_w  # center x
    cy = (y + h / 2) / frame_h
    nw = w / frame_w
    nh = h / frame_h

    # Direction estimation
    if cx < DIRECTION_LEFT:
        direction = "left"
    elif cx > DIRECTION_RIGHT:
        direction = "right"
    else:
        direction = "center"

    # Distance estimation (based on bbox area ratio)
    area_ratio = nw * nh
    if area_ratio > DISTANCE_NEAR:
        distance = "near"
    elif area_ratio > DISTANCE_MEDIUM:
        distance = "medium"
    else:
        distance = "far"

    is_hazard = cls_name in HAZARD_CLASSES

    return {
        "c": cls_name,
        "x": round(cx, 3),
        "y": round(cy, 3),
        "w": round(nw, 3),
        "h": round(nh, 3),
        "p": round(conf, 3),
        "d": direction,
        "dist": distance,
        "hz": is_hazard,
    }


# ─────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────
def main_loop():
    """Run continuous YOLO inference and output results over serial."""
    print("[GuideLens] Starting K210 GuideLens Detection")
    print("[GuideLens] Protocol: JSON lines over USB serial")

    init_camera()
    task = load_model()
    if task is None:
        print("[GuideLens] FATAL: No model — entering idle loop")
        set_led((255, 0, 0))
        while True:
            time.sleep(1)

    set_led(LED_IDLE_COLOR)

    frame_num = 0
    fps_counter = 0
    fps_timer = time.ticks_ms()
    current_fps = 0.0

    print("[GuideLens] Detection loop starting...")

    while True:
        try:
            # Capture frame
            img = sensor.snapshot()
            frame_num += 1
            fps_counter += 1

            # Run YOLO inference on KPU
            code = kpu.run_yolo2(task, img)

            # Process detections
            objects = []
            has_hazard = False

            if code:
                for det in code:
                    # KPU detection: (x, y, w, h, class_id, prob)
                    # Actually returns objects with .rect(), .classid(), .value()
                    obj_info = analyse_detection(
                        (det.rect()[0], det.rect()[1], det.rect()[2], det.rect()[3],
                         det.classid(), det.value()),
                        CAMERA_WIDTH, CAMERA_HEIGHT
                    )
                    objects.append(obj_info)
                    if obj_info["hz"]:
                        has_hazard = True

                    # Draw bounding box on image (for JPEG preview)
                    color = (255, 0, 0) if obj_info["hz"] else (0, 255, 0)
                    img.draw_rectangle(
                        det.rect()[0], det.rect()[1],
                        det.rect()[2], det.rect()[3],
                        color=color, thickness=2
                    )
                    # Draw label
                    label = "{} {:.0f}%".format(obj_info["c"], obj_info["p"] * 100)
                    img.draw_string(
                        det.rect()[0], max(0, det.rect()[1] - 12),
                        label, color=color, scale=1
                    )

            # Update LED based on detections
            if has_hazard:
                set_led(LED_HAZARD_COLOR)
            elif objects:
                set_led(LED_DETECTION_COLOR)
            else:
                set_led(LED_IDLE_COLOR)

            # Send detection JSON
            if objects or frame_num % 30 == 0:  # Send even empty frames periodically
                send_json({
                    "t": "det",
                    "f": frame_num,
                    "objs": objects,
                })

            # Send JPEG frame at configured interval
            if SEND_JPEG_FRAMES and frame_num % JPEG_SEND_INTERVAL == 0:
                send_jpeg(img, frame_num)

            # Calculate FPS every second
            elapsed = time.ticks_diff(time.ticks_ms(), fps_timer)
            if elapsed >= 1000:
                current_fps = fps_counter * 1000.0 / elapsed
                fps_counter = 0
                fps_timer = time.ticks_ms()

                # Send heartbeat with FPS and memory
                send_json({
                    "t": "hb",
                    "f": frame_num,
                    "fps": round(current_fps, 1),
                    "mem": gc.mem_free(),
                })

                # Garbage collect periodically
                gc.collect()

        except MemoryError:
            gc.collect()
            send_error("MemoryError — gc.collect() called")
            time.sleep_ms(100)

        except Exception as e:
            send_error(str(e))
            time.sleep_ms(50)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("[GuideLens] Stopped by user")
    except Exception as e:
        print("[GuideLens] Fatal error:", e)
        set_led((255, 0, 0))
    finally:
        try:
            kpu.deinit(task)
        except Exception:
            pass

# M5Stack K210 GuideLens — Configuration
# ========================================
# Adjust these settings for your environment.

# ─────────────────────────────────────────
# Camera
# ─────────────────────────────────────────
CAMERA_WIDTH = 224       # Input resolution for YOLO (must be 224 for the kmodel)
CAMERA_HEIGHT = 224
CAMERA_FPS = 15          # Target capture FPS

# ─────────────────────────────────────────
# YOLO Model
# ─────────────────────────────────────────
# Path to the .kmodel file.
# If using SD card: "/sd/yolo_20class.kmodel"
# If flashed to internal storage: "/flash/yolo_20class.kmodel"
MODEL_PATH = "/sd/yolo_20class.kmodel"
MODEL_FLASH_PATH = "/flash/yolo_20class.kmodel"

# Detection confidence threshold (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.45

# Non-max suppression IoU threshold
NMS_THRESHOLD = 0.3

# ─────────────────────────────────────────
# YOLO 20-class labels (COCO subset)
# ─────────────────────────────────────────
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv"
]

# Classes that are hazardous for navigation
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorbike", "bus",
    "truck", "dog", "horse", "chair", "boat"
}

# ─────────────────────────────────────────
# UART Serial Output
# ─────────────────────────────────────────
UART_BAUDRATE = 115200   # Baud rate for UART communication
UART_TX_PIN = None       # None = use USB serial (default)
UART_RX_PIN = None

# Send JPEG frames over serial (for live preview on host)
SEND_JPEG_FRAMES = True
JPEG_QUALITY = 40        # 0-100, lower = smaller/faster
JPEG_SEND_INTERVAL = 3   # Send JPEG every N frames (to save bandwidth)

# ─────────────────────────────────────────
# LED Status Indicators
# ─────────────────────────────────────────
LED_ENABLED = True
LED_HAZARD_COLOR = (255, 0, 0)      # Red for hazard
LED_DETECTION_COLOR = (0, 255, 0)   # Green for normal detection
LED_IDLE_COLOR = (0, 0, 50)         # Dim blue when idle

# ─────────────────────────────────────────
# Proximity / Hazard Analysis
# ─────────────────────────────────────────
# Bbox area as fraction of frame area
DISTANCE_NEAR = 0.15      # >15% of frame → near
DISTANCE_MEDIUM = 0.05    # >5% of frame → medium
# else → far

# Direction thresholds (fraction of frame width)
DIRECTION_LEFT = 0.33     # bbox center < 33% → left
DIRECTION_RIGHT = 0.66    # bbox center > 66% → right
# else → center

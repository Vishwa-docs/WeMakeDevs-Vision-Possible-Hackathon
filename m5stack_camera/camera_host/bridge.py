"""
K210 Camera Host Bridge
========================
Receives YOLO detections + JPEG frames from the K210 over USB serial,
and serves them via WebSocket to the standalone camera UI.

Runs on your laptop. Completely independent of the main WorldLens backend.

Usage:
    python bridge.py --port /dev/tty.usbserial-XXXX
    python bridge.py --auto   # auto-detect K210 serial port

WebSocket:  ws://localhost:8001/ws
HTTP UI:    http://localhost:8001
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("k210_bridge")

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    logger.error("pyserial not installed. Run: pip install pyserial")
    sys.exit(1)

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve
except ImportError:
    logger.error("websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    from aiohttp import web
except ImportError:
    logger.error("aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)


# ─────────────────────────────────────────
# Serial reader — reads K210 JSON protocol
# ─────────────────────────────────────────
class K210SerialReader:
    """
    Reads line-based JSON protocol from the K210 over USB serial.

    Protocol:
      {"t":"det",...}   → detection data
      {"t":"jpg","sz":N} + N raw bytes → JPEG frame
      {"t":"hb",...}    → heartbeat
      {"t":"err",...}   → error
    """

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial: serial.Serial | None = None
        self._running = False

        # Latest state
        self.latest_detections: list[dict] = []
        self.latest_frame_jpeg: bytes = b""
        self.frame_number: int = 0
        self.fps: float = 0.0
        self.memory_free: int = 0
        self.connected: bool = False
        self._last_data_time: float = 0

    def open(self) -> bool:
        """Open the serial connection to the K210."""
        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                timeout=1.0,
                write_timeout=1.0,
            )
            self.connected = True
            logger.info("Connected to K210 on %s @ %d baud", self.port, self.baudrate)
            return True
        except serial.SerialException as e:
            logger.error("Failed to open %s: %s", self.port, e)
            return False

    def close(self):
        """Close the serial connection."""
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
        logger.info("Serial connection closed")

    async def read_loop(self):
        """
        Continuously read and parse JSON lines from the K210.
        Runs in the asyncio event loop using run_in_executor for blocking reads.
        """
        self._running = True
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Read one line (blocking, in thread pool)
                line = await loop.run_in_executor(None, self._read_line)
                if line is None:
                    await asyncio.sleep(0.01)
                    continue

                self._last_data_time = time.time()
                self._parse_line(line)

            except serial.SerialException as e:
                logger.error("Serial error: %s", e)
                self.connected = False
                await asyncio.sleep(1.0)
                # Try to reconnect
                if self.open():
                    logger.info("Reconnected to K210")
                    continue
                break

            except Exception as e:
                logger.warning("Read error: %s", e)
                await asyncio.sleep(0.1)

    def _read_line(self) -> str | None:
        """Read one line from serial (blocking)."""
        if not self._serial or not self._serial.is_open:
            return None
        try:
            raw = self._serial.readline()
            if raw:
                return raw.decode("utf-8", errors="replace").strip()
        except (serial.SerialException, UnicodeDecodeError):
            pass
        return None

    def _read_bytes(self, n: int) -> bytes:
        """Read exactly n bytes from serial (blocking)."""
        if not self._serial:
            return b""
        return self._serial.read(n)

    def _parse_line(self, line: str):
        """Parse a JSON line from the K210 protocol."""
        if not line or not line.startswith("{"):
            return

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return

        msg_type = data.get("t", "")

        if msg_type == "det":
            self.latest_detections = data.get("objs", [])
            self.frame_number = data.get("f", self.frame_number)

        elif msg_type == "jpg":
            # Next N bytes are raw JPEG
            size = data.get("sz", 0)
            self.frame_number = data.get("f", self.frame_number)
            if size > 0 and size < 500_000:  # sanity limit
                try:
                    loop = asyncio.get_event_loop()
                    jpg_data = self._read_bytes(size)
                    if len(jpg_data) == size:
                        self.latest_frame_jpeg = jpg_data
                except Exception:
                    pass

        elif msg_type == "hb":
            self.fps = data.get("fps", 0.0)
            self.memory_free = data.get("mem", 0)
            self.frame_number = data.get("f", self.frame_number)

        elif msg_type == "err":
            logger.warning("K210 error: %s", data.get("msg", "unknown"))

    def get_state(self) -> dict:
        """Get the current bridge state as a JSON-serializable dict."""
        return {
            "connected": self.connected,
            "frame_number": self.frame_number,
            "fps": self.fps,
            "memory_free": self.memory_free,
            "detections": self.latest_detections,
            "has_frame": len(self.latest_frame_jpeg) > 0,
            "last_data_age": round(time.time() - self._last_data_time, 1) if self._last_data_time else -1,
        }


# ─────────────────────────────────────────
# WebSocket server — broadcasts to UI clients
# ─────────────────────────────────────────
class BridgeServer:
    """
    Serves:
      - HTTP on port 8001 (static camera UI)
      - WebSocket on ws://localhost:8001/ws (detection + frame stream)
    """

    def __init__(self, reader: K210SerialReader, host: str = "0.0.0.0", port: int = 8001):
        self.reader = reader
        self.host = host
        self.port = port
        self._ws_clients: set = set()
        self._app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Set up HTTP routes."""
        static_dir = Path(__file__).parent / "static"
        self._app.router.add_get("/", self._serve_index)
        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/status", self._status_handler)
        self._app.router.add_get("/frame.jpg", self._frame_handler)
        if static_dir.exists():
            self._app.router.add_static("/static", static_dir)

    async def _serve_index(self, request):
        """Serve the standalone camera UI."""
        index_path = Path(__file__).parent / "static" / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(
            text="<h1>K210 Camera Bridge</h1><p>Static files not found.</p>",
            content_type="text/html",
        )

    async def _status_handler(self, request):
        """REST endpoint for bridge status."""
        return web.json_response(self.reader.get_state())

    async def _frame_handler(self, request):
        """Serve the latest JPEG frame."""
        if self.reader.latest_frame_jpeg:
            return web.Response(
                body=self.reader.latest_frame_jpeg,
                content_type="image/jpeg",
            )
        # Return a 1x1 transparent pixel if no frame
        return web.Response(status=204)

    async def _ws_handler(self, request):
        """WebSocket handler — streams detections to connected UI clients."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        logger.info("WebSocket client connected (%d total)", len(self._ws_clients))

        try:
            async for msg in ws:
                # Client can send commands (future: control camera settings)
                if msg.type == web.WSMsgType.TEXT:
                    logger.debug("WS received: %s", msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            self._ws_clients.discard(ws)
            logger.info("WebSocket client disconnected (%d remaining)", len(self._ws_clients))

        return ws

    async def _broadcast_loop(self):
        """Periodically broadcast detection state to all WebSocket clients."""
        import base64

        while True:
            if self._ws_clients:
                state = self.reader.get_state()

                # Include base64 JPEG frame if available
                if self.reader.latest_frame_jpeg:
                    state["frame_b64"] = base64.b64encode(
                        self.reader.latest_frame_jpeg
                    ).decode("ascii")

                msg = json.dumps(state)
                # Send to all connected clients
                dead = set()
                for ws in self._ws_clients:
                    try:
                        await ws.send_str(msg)
                    except Exception:
                        dead.add(ws)
                self._ws_clients -= dead

            await asyncio.sleep(0.1)  # ~10 updates/sec to UI

    async def start(self):
        """Start the HTTP + WebSocket server and broadcast loop."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(
            "Bridge server running on http://%s:%d",
            self.host, self.port,
        )
        logger.info("Camera UI: http://localhost:%d", self.port)
        logger.info("WebSocket:  ws://localhost:%d/ws", self.port)

        # Start broadcast loop
        asyncio.create_task(self._broadcast_loop())


# ─────────────────────────────────────────
# Auto-detect K210 serial port
# ─────────────────────────────────────────
def find_k210_port() -> str | None:
    """Try to auto-detect the K210 USB serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        mfr = (p.manufacturer or "").lower()
        # K210 UnitV typically shows up as FTDI or M5Stack
        if any(kw in desc for kw in ("ftdi", "m5stack", "usb serial", "unitv")):
            logger.info("Auto-detected K210 on %s (%s)", p.device, p.description)
            return p.device
        if any(kw in mfr for kw in ("ftdi", "m5stack")):
            logger.info("Auto-detected K210 on %s (mfr: %s)", p.device, mfr)
            return p.device

    # Fallback: try common macOS/Linux patterns
    patterns = [
        "/dev/tty.usbserial-*",
        "/dev/ttyUSB*",
        "/dev/tty.SLAB_USBtoUART*",
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            logger.info("Found serial port via glob: %s", matches[0])
            return matches[0]

    return None


# ─────────────────────────────────────────
# Demo mode — generates fake detections for UI testing
# ─────────────────────────────────────────
class DemoReader:
    """
    Fake K210 reader for testing the UI without hardware.
    Generates synthetic detections.
    """

    def __init__(self):
        self.latest_detections = []
        self.latest_frame_jpeg = b""
        self.frame_number = 0
        self.fps = 15.0
        self.memory_free = 2_000_000
        self.connected = True
        self._last_data_time = time.time()

    async def read_loop(self):
        """Generate fake detections."""
        import math
        while True:
            t = time.time()
            self.frame_number += 1
            self._last_data_time = t

            # Simulate a person walking left-right
            cx = 0.3 + 0.4 * math.sin(t * 0.5)
            self.latest_detections = [
                {
                    "c": "person",
                    "x": round(cx, 3),
                    "y": 0.5,
                    "w": 0.15,
                    "h": 0.4,
                    "p": 0.92,
                    "d": "left" if cx < 0.33 else ("right" if cx > 0.66 else "center"),
                    "dist": "medium",
                    "hz": True,
                },
            ]

            # Occasionally add a car
            if int(t) % 5 == 0:
                self.latest_detections.append({
                    "c": "car",
                    "x": 0.7,
                    "y": 0.6,
                    "w": 0.25,
                    "h": 0.2,
                    "p": 0.85,
                    "d": "right",
                    "dist": "far",
                    "hz": True,
                })

            await asyncio.sleep(1.0 / 15)

    def open(self):
        return True

    def close(self):
        pass

    def get_state(self):
        return {
            "connected": True,
            "frame_number": self.frame_number,
            "fps": self.fps,
            "memory_free": self.memory_free,
            "detections": self.latest_detections,
            "has_frame": False,
            "last_data_age": round(time.time() - self._last_data_time, 1),
            "demo_mode": True,
        }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="K210 Camera Host Bridge")
    parser.add_argument("--port", type=str, help="Serial port (e.g., /dev/tty.usbserial-XXXX)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect K210 serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--http-port", type=int, default=8001, help="HTTP/WebSocket port")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no hardware)")
    args = parser.parse_args()

    # Determine reader (real hardware or demo)
    if args.demo:
        logger.info("Running in DEMO mode (no K210 hardware needed)")
        reader = DemoReader()
    else:
        # Find serial port
        port = args.port
        if args.auto or not port:
            port = find_k210_port()
            if not port:
                logger.error(
                    "Could not auto-detect K210 serial port. "
                    "Use --port /dev/tty.usbserial-XXXX or --demo for testing."
                )
                sys.exit(1)

        reader = K210SerialReader(port, args.baud)
        if not reader.open():
            logger.error("Failed to open serial port. Use --demo for testing.")
            sys.exit(1)

    # Start bridge server
    server = BridgeServer(reader, host=args.host, port=args.http_port)
    await server.start()

    # Start serial read loop
    logger.info("Starting K210 read loop...")
    await reader.read_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bridge stopped by user")

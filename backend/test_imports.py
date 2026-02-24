"""Quick import test for Day 1 dependencies."""
try:
    from vision_agents.core import Agent, AgentLauncher, Runner, User
    from vision_agents.plugins import gemini, getstream
    print("OK: Vision Agents SDK imports")
    print(f"  Agent: {Agent}")
    print(f"  Gemini.Realtime: {gemini.Realtime}")
    print(f"  getstream.Edge: {getstream.Edge}")
except ImportError as e:
    print(f"FAIL: {e}")

try:
    import cv2
    print(f"OK: OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"FAIL: {e}")

try:
    import aiosqlite
    print("OK: aiosqlite")
except ImportError as e:
    print(f"FAIL: {e}")

try:
    import httpx
    print(f"OK: httpx {httpx.__version__}")
except ImportError as e:
    print(f"FAIL: {e}")

print("\nAll Day 1 imports verified!")

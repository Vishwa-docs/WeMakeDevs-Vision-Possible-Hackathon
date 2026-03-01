# boot.py — M5Stack K210 GuideLens Boot Script
# ==============================================
# This file runs automatically when the K210 powers on.
# It sets up hardware and then imports main.py.

import gc

# Free memory before loading the heavy model
gc.collect()

# Set CPU frequency to maximum for best inference speed
try:
    import machine
    machine.freq(400000000)  # 400 MHz
except Exception:
    pass

# Run the main GuideLens detection loop
try:
    import main
except Exception as e:
    # If main fails, blink LED red to indicate error
    try:
        from modules import ws2812
        led = ws2812(8, 1)
        import time
        for _ in range(10):
            led.set_led(0, (255, 0, 0))
            led.display()
            time.sleep_ms(300)
            led.set_led(0, (0, 0, 0))
            led.display()
            time.sleep_ms(300)
    except Exception:
        pass
    print("BOOT ERROR:", e)

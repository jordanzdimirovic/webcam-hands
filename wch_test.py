import webcam_hands as wch
import numpy as np
import sys
import time
print(wch.WebcamHands.get_default_options())
app = wch.WebcamHands(
    options = {
        "view_camera": True
    }
)

app.start()

print("App started!")

while True:
    if app.COMMS["RH_in_frame"]:
        print(f"D {np.round((app.get_palm_position(1, 0) - app.get_palm_position(1, 1)) / app.tracking_snapshots_deltatime(1, 0, 1), 1)}" + " " * 10, end = "\r")
        sys.stdout.flush()
        # | V {round(np.sum((app.get_palm_position(1, 1) - app.get_palm_position(1, 0))**2) * 100, 1)}
        if not app.COMMS["running"]: break
    else:
        time.sleep(0.05)
import webcam_hands as wch
import numpy as np
import sys
import time
app = wch.WebcamHands("options.json")

app.start()

print("App started!")

while True:
    #if not app.COMMS["running"]: break
    #print(f"Gestures: {', '.join(x.name for x in app.BUFFERS['RH_GESTURES'])}")
    print(f"Gestures: {app.BUFFERS['RH_GESTURES'][0].name}")
    # print(f"D {np.round((app.get_palm_position(1, 0) - app.get_palm_position(1, 1)) / app.tracking_snapshots_deltatime(1, 0, 1), 1)}" + " " * 10, end = "\r")
    # sys.stdout.flush()
    # | V {round(np.sum((app.get_palm_position(1, 1) - app.get_palm_position(1, 0))**2) * 100, 1)}
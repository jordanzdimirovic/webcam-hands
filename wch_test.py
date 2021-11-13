import webcam_hands as wch

app = wch.WebcamHands(
    options = {
        "view_camera": False,
        "flip_camera": True
    }
)

app.start()

while True:
    x = input().lower().strip()
    if x == 'q':
        app.COMMS["running"] = False
        break
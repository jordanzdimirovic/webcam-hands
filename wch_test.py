import webcam_hands as wch

app = wch.WebcamHands(
    options = {
        "view_camera": True
    }
)

app.start()
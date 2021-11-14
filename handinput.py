# Control computer with hand input
# Imports
from webcam_hands import WebcamHands
import vectorfuncs as vect
import numpy as np
from math import inf as infinity
# Computer input libraries (windows)
import win32api as winapi, win32con as wincon

# CONSTANTS
MOUSE_SENSITIVITY = 600
MIN_VELOCITY_GATE = 0.0035
from time import sleep
def detect_click():
    raise NotImplementedError()
LEFT_CLICK_ACTIVATED = False
RIGHT_CLICK_ACTIVATED = False
def main():
    # Create the API instance
    app = WebcamHands(options = "options.json")
    # Start it
    app.start()
    print("Done!")
    # Get the current mouse position and store it as nparray
    
    mouse_position = np.array(list(map(float, winapi.GetCursorPos())))
    def do_clicks():
        global LEFT_CLICK_ACTIVATED
        global RIGHT_CLICK_ACTIVATED
        # LEFT CLICK
        if app.current("RH_GESTURES").id == app.gesture("middle_up"):
            if not LEFT_CLICK_ACTIVATED:
                LEFT_CLICK_ACTIVATED = True
                winapi.mouse_event(wincon.MOUSEEVENTF_LEFTDOWN, i_mp[0], i_mp[1], 0, 0)
        else:
            if LEFT_CLICK_ACTIVATED:
                LEFT_CLICK_ACTIVATED = False
                winapi.mouse_event(wincon.MOUSEEVENTF_LEFTUP, i_mp[0], i_mp[1], 0, 0)
        ##############
        # RIGHT CLICK
        if app.current("RH_GESTURES").id == app.gesture("index_up"):
            if not RIGHT_CLICK_ACTIVATED:
                RIGHT_CLICK_ACTIVATED = True
                winapi.mouse_event(wincon.MOUSEEVENTF_RIGHTDOWN, i_mp[0], i_mp[1], 0, 0)
        else:
            if RIGHT_CLICK_ACTIVATED:
                RIGHT_CLICK_ACTIVATED = False
                winapi.mouse_event(wincon.MOUSEEVENTF_RIGHTUP, i_mp[0], i_mp[1], 0, 0)
        ##############  
    try:
        while True:
            sleep(0.0001)
            if app.COMMS["RH_in_frame"]:
                if app.current("RH_GESTURES").id != app.gesture("fist"):
                    # Get the velocity of righthand
                    vel = vect.gate(app.velocity[1], MIN_VELOCITY_GATE)
                    # Move the mouse by the velocity
                    mouse_position[0] += vel[0] * MOUSE_SENSITIVITY
                    mouse_position[1] += vel[1] * MOUSE_SENSITIVITY
                    # Clamp the position in the screen
                    vect.clamp(mouse_position, (-1920,0), (2560, 1440))
                    i_mp = mouse_position.astype(int)
                    winapi.SetCursorPos(i_mp)
                    do_clicks()

            else:
                sleep(0.05)
    
    except KeyboardInterrupt:
        app.COMMS["running"] = False
        quit()

if __name__ == '__main__': main()
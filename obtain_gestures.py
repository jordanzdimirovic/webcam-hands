"""
Script to perform the following tasks:
- Detect the position of hands on the screen
- Allow for the black and white image to be stored in the respective folder
"""
# IMPORTS
import cv2
from os import path
import numpy as np

# CONSTANTS
DATA_FOLDER = path.join("dataraw", "raw", "gestures")

def main():
    # Init cam device
    cam = cv2.VideoCapture(0)
    
    while True:
        # Read a single frame
        _, frame = cam.read()

        # Convert the frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Show the image
        cv2.imshow("Camera", frame)
        
        # Escape -> break from loop
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
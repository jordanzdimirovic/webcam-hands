"""
DEPRECATED
Script to perform the following tasks:
- Detect the position of hands on the screen
- Allow for the black and white image to be stored in the respective folder

CONSIDER MEDIA PIPE!!!!!!!!!!!!
"""
# IMPORTS
import cv2
from os import path
import numpy as np

# CONSTANTS
DATA_DIR = "data"
HAAR_CASC_DIR = path.join(DATA_DIR, "haarcascade")
GEST_DATA_DIR = path.join(DATA_DIR, "raw", "gestures")

def main():
    # Init cam device
    cam = cv2.VideoCapture(0)
    
    # Create a new Haar-Cascade classifier from the cascade XML
    hand_cascade = cv2.CascadeClassifier(path.join(HAAR_CASC_DIR, "hand_class.xml"))
    hand_cascade2 = cv2.CascadeClassifier(path.join(HAAR_CASC_DIR, "palm.xml"))
    hand_cascade3 = cv2.CascadeClassifier(path.join(HAAR_CASC_DIR, "fist.xml"))
    while True:
        # Read a single frame
        _, frame = cam.read()
        
        # Convert the frame to grayscale
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_frame = np.zeros(frame.shape)
        norm_frame = cv2.normalize(frame, norm_frame, 0, 255, cv2.NORM_MINMAX)
        # Cascade
        hands_det = hand_cascade.detectMultiScale(
            frame,
            1.19,
            10
        )

        hands_det2 = hand_cascade2.detectMultiScale(
            frame,
            1.1,
            2
        )

        hands_det3 = hand_cascade3.detectMultiScale(
            frame,
            1.1,
            2
        )

        for x,y,l,w in hands_det:
            cv2.rectangle(frame, (x, y), (x+w, y+l), (255, 0, 0)) # Blue

        for x,y,l,w in hands_det2:
            cv2.rectangle(frame, (x, y), (x+w, y+l), (0, 0, 0)) # White

        for x,y,l,w in hands_det3:
            cv2.rectangle(frame, (x, y), (x+w, y+l), (0, 0, 255)) # Red

        # Show the image
        #cv2.imshow("Camera norm", norm_frame)
        cv2.imshow("Camera", frame)
        # Escape -> break from loop
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
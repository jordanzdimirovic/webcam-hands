"""
Obtain hand landmark data, using Google's "MediaPipe" API.
"""
# Imports
import cv2
import numpy as np
import json
import re
import mediapipe as mp
import os
from timeit import default_timer as timer

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

### CONSTANTS
# Hand landmark detection model parameters
HANDS_DET_CONF = 0.5
HANDS_MOD_COMPLEX = 0
HANDS_TRACK_CONF = 0.5

# Keycodes:
KEYCODE_ESC = 27
KEYCODE_SPACE = 32

# Other:
TIME_BETWEEN_HAND_CAPTURES = 0.3

MSG_NORM_POSITION = (0.5, 0.85) 
MSG_FONT = cv2.FONT_HERSHEY_SIMPLEX
MSG_FONTSIZE = 0.6
MSG_CAPTURING_TRUE = "Currently Capturing! Press SPACE to stop."
MSG_CAPTURING_TRUE_COLOUR = (20, 255, 30)

MSG_CAPTURING_FALSE = "Not Capturing. Press SPACE to begin."
MSG_CAPTURING_FALSE_COLOUR = (20, 30, 255)

# SCRIPT
def main():
    folder = input("What are we classifying? : ")

    # Determine the save locations
    save_folder = "data/gesture_data/" + folder
    save_file = "data/gesture_data/" + folder + "/data.csv"

    # Create required directories, if they don't exist
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    
    def save_data(hands_res):
        """
        Store the handedness and landmark data of both hands in CSV format.
        Unfortunately, the data is in a strange format, and cannot be extracted easily.
        TODO: Determine whether there is an easier way to perform this. StackOverflow?
        """
        for hand_index in range(len(hands_res.multi_hand_landmarks)):
            handedness = ('0','1')[re.findall("(Right)|(Left)", str(hands_res.multi_handedness[hand_index]))[0][0] == "Right"]
            # Append to CSV: handedness,x1,y1,z1,x2,...
            values_to_write = [handedness] + re.findall(r"[0-9]+\.[0-9]*", str(hands_res.multi_hand_landmarks[hand_index]))
            with open(save_file, 'a') as fileobj:
                fileobj.write("\n" + ",".join(values_to_write))

    # Establish webcam connection
    cam = cv2.VideoCapture(0)
    
    # Get some properties of the camera (CONSTANTS)
    CAM_FRAME_HEIGHT = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    CAM_FRAME_WIDTH = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Create mediapipe hands classifier
    hands_mdl = mp_hands.Hands(
        model_complexity = HANDS_MOD_COMPLEX,
        min_detection_confidence = HANDS_DET_CONF,
        min_tracking_confidence = HANDS_TRACK_CONF
    )

    # Is the program currently storing hand feature data? (this is a toggle)
    # Note that the program will store data every TIME_BETWEEN_HAND_CAPTURES seconds
    is_storing_features = False
    when_last_stored_features = 0

    # Place to store current number of hand caps
    n_caps = 0

    # Main webcam loop
    while True:
        # Read a single frame
        success, frame = cam.read()
        
        if not success:
            print("Camera didn't publish a frame. Continuing...")
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get bounds of text
        textsize = cv2.getTextSize(MSG_CAPTURING_TRUE if is_storing_features else MSG_CAPTURING_FALSE, MSG_FONT, MSG_FONTSIZE, 2)[0]
        text_offset_x = textsize[0] // 2
        text_offset_y = textsize[1] // 2

        # Write the text to the frame
        frame = cv2.putText(
            frame,
            MSG_CAPTURING_TRUE if is_storing_features else MSG_CAPTURING_FALSE,
            (int(MSG_NORM_POSITION[0] * CAM_FRAME_WIDTH - text_offset_x), int(MSG_NORM_POSITION[1] * CAM_FRAME_HEIGHT + text_offset_y)),
            MSG_FONT,
            MSG_FONTSIZE,
            MSG_CAPTURING_TRUE_COLOUR if is_storing_features else MSG_CAPTURING_FALSE_COLOUR,
            2
        )        

        hands_res = hands_mdl.process(frame)
        if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
            for hand_index in range(len(hands_res.multi_hand_landmarks)):
                landmarks = hands_res.multi_hand_landmarks[hand_index]      
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.imshow("Hand Gestures - Generate Data", frame)
        
        if is_storing_features and timer() - when_last_stored_features >= TIME_BETWEEN_HAND_CAPTURES:
            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                # Store last time as current time
                when_last_stored_features = timer()
                # Store the data
                save_data(hands_res)
                # Print that we saved data
                n_caps += 1
                print(f"HAND CAPTURED. ID: {n_caps}")

        if cv2.waitKey(1) == KEYCODE_SPACE:
            is_storing_features = not is_storing_features

        if cv2.waitKey(1) == KEYCODE_ESC:
            break
    
    cv2.destroyAllWindows()
    cam.release()
        
        

if __name__ == '__main__':
    main()

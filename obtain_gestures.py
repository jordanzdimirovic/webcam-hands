"""
Obtain hand landmark data, using Google's "MediaPipe" API.
"""
# Imports
import cv2
import numpy as np

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# CONSTANTS
HANDS_DET_CONF = 0.5
HANDS_MOD_COMPLEX = 0
HANDS_TRACK_CONF = 0.5
def main():
    folder = input("What are we classifying? : ")
    save_index = 0
    save_location = "data/gestures/" + folder + "/data.csv"

    def save_data(hands_res):
        if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
            for hand_index in hands_res.multi_hand_landmarks:
                handedness = hands_res.multi_handedness[hand_index]
                for datapoint in hands_res.multi_hand_landmarks:

    # Establish webcam connection
    cam = cv2.VideoCapture(0)
    # Create mediapipe hands classifier
    hands_mdl = mp_hands.Hands(
        model_complexity = HANDS_MOD_COMPLEX,
        min_detection_confidence = HANDS_DET_CONF,
        min_tracking_confidence = HANDS_TRACK_CONF
    )

    while True:
        # Read a single frame
        success, frame = cam.read()
        
        if not success:
            print("Camera didn't publish a frame. Continuing...")
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hands_res = hands_mdl.process(frame)
        if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
            for hand_index in hands_res.multi_hand_landmarks:
                landmarks = hands_res.multi_hand_landmarks[hand_index]
                handedness = hands_res.multi_handedness[hand_index]
                print(handedness)
                #print(hand)
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        cv2.imshow("Hands!", frame)

        if cv2.waitKey(1) == 13:
            save_data(hands_res)

        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()
    cam.release()
        
        

if __name__ == '__main__':
    main()

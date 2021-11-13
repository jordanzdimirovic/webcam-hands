"""
Webcam handtracking / gesture API.
"""
trained_gestures = "open_palm peace fist middle_forwardfacing".split()

import cv2
import tensorflow as tf
import keras
import sys
import numpy as np
from timeit import default_timer as timer
from dataclasses import dataclass
import mediapipe as mp
import time
from collections import deque

from threading import Thread, Event

import re
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from copy import deepcopy

@dataclass
class Gesture():
    id: int
    timestamp: float
    @property
    def name(self):
        return "" if self.id == -1 else trained_gestures[self.id]

class WebcamHands():
    """
    Faciliates hand tracking and gesture classification through the webcam.
    """
    # Class-level methods
    def get_default_options():
        """
        Return a new instance of the default WebcamHands options dictionary.
        """
        return {
            "gesture_classifier_model_path": "gesture_model",
            "flip_camera": False,
            "view_camera": True,
            "video_device_index": 0,
            "landmark_model_complexity": 0,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        }

    def __init__(self, options = None):
        # Create default options
        self.options = type(self).get_default_options()
        self.COMMS = {"running": True, "tracking_framerate": 0, "gestures_pending": [], "LH_in_frame": False, "RH_in_frame": False}
        self.BUFFERS = {
            "LH_LANDMARKS": None,
            "RH_LANDMARKS": None,
            "LH_GESTURES": None,
            "RH_GESTURES": None
        }
        # Set all options, provided they're valid
        if options:
            for option in options:
                if option in self.options:
                    self.options[option] = options[option]            
                else:
                    raise ValueError(f"Option '{option}' is not valid.\nFor valid options and their defaults, refer to WebcamHands.get_default_options")

    def __THREAD_landmark_tracking(self, EVENT_CLASSIFY, LEFTHAND_BUFFER: np.array, RIGHTHAND_BUFFER: np.array):
        """
            MediaPipe hand tracking thread, created by main_manager thread.
            Responsible for:
            - Handedness classification
            - Hand landmark location prediction
        """
        PREVIEW_WINDOW_NAME = "Hand-Tracking Preview"
        
        # Get camera device
        cam = cv2.VideoCapture(self.options["video_device_index"])

        # Initialise mediapipe hands classifier
        hands_mdl = mp_hands.Hands(
            model_complexity = self.options["landmark_model_complexity"],
            min_detection_confidence = self.options["min_detection_confidence"],
            min_tracking_confidence = self.options["min_tracking_confidence"]
        )
        
        last_frame = timer()
        # Main loop
        while self.COMMS["running"]:
            # Calculate frame rate
            curr_frame = timer()
            self.COMMS["tracking_framerate"] = 1/(curr_frame - last_frame)
            last_frame = curr_frame

            success, frame = cam.read()
            if not success:
                print("Camera didn't publish a frame. Continuing...")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if self.options["flip_camera"]:
                frame = cv2.flip(frame, 1)

            hands_res = hands_mdl.process(frame)

            hands_in_frame = set()
            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                for hand_index in range(len(hands_res.multi_hand_landmarks)):
                    landmarks = hands_res.multi_hand_landmarks[hand_index]
                    handedness = (0,1)[re.findall("(Right)|(Left)", str(hands_res.multi_handedness[hand_index]))[0][0] == "Right"]
                    
                    hands_in_frame.add(handedness)

                    self.COMMS["gestures_pending"].append(handedness)
                    
                    # Get the target buffer to write to
                    target_buffer = RIGHTHAND_BUFFER if handedness else LEFTHAND_BUFFER
                    for i, landmark in enumerate(landmarks.landmark):
                        target_buffer[i] = [landmark.x, landmark.y, landmark.z]

                    # Draw, if options are set
                    if self.options["view_camera"]:
                        mp_drawing.draw_landmarks(
                            frame,
                            landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                EVENT_CLASSIFY.set()

            self.COMMS["RH_in_frame"] = (1 in hands_in_frame)
            self.COMMS["LH_in_frame"] = (0 in hands_in_frame)


            cv2.imshow(PREVIEW_WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                self.COMMS["running"] = False

        cv2.destroyAllWindows()
                        

    def __THREAD_gesture_classification(self, EVENT_CLASSIFY, LEFTHAND_GESTURE_BUFFER: deque, RIGHTHAND_GESTURE_BUFFER: deque, LEFTHAND_BUFFER: np.array, RIGHTHAND_BUFFER: np.array):
        """
            Tensorflow model gesture classifier thread, created by main_manager thread.
            Responsible for:
            - Hand gesture classification (data obtained from LEFTHAND_BUFFER and RIGHTHAND_BUFFER)
        """
        # Load the model
        model_path = self.options["gesture_classifier_model_path"]
        gesture_model = keras.models.load_model(model_path)

        # Wait for the event to occur
        while self.COMMS["running"]:
            if EVENT_CLASSIFY.wait(5):
                pending = self.COMMS["gestures_pending"]
                for hand in pending:
                    selected_buffer = RIGHTHAND_BUFFER if hand else LEFTHAND_BUFFER
                    # Unnest the buffer data, appending hand to the beginning
                    features = selected_buffer.flatten()
                    data_for_mdl = np.array([np.insert(features, 0, hand)])
                    # Predict the gesture
                    gesture_predicted = np.argmax(gesture_model.predict(data_for_mdl)[0])
                    # Push it into the correct deque, pop from the end
                    selected_gesture_deque = RIGHTHAND_GESTURE_BUFFER if hand else LEFTHAND_GESTURE_BUFFER
                    
                    if gesture_predicted != selected_gesture_deque[0].id:
                        selected_gesture_deque.appendleft(
                            Gesture(
                                id = gesture_predicted,
                                timestamp = timer()
                            )
                        )

                        selected_gesture_deque.pop()

                # Clear the pending list
                self.COMMS["gestures_pending"].clear()
                # Clear the event
                EVENT_CLASSIFY.clear()



    def __THREAD_main_manager(self):
        """
            Main thread, created by runtime.
            Responsible for:
            - Managing buffers to facilitate thread communication
            - Create and run the landmark_tracking and gesture_classification threads
        """
        LEFTHAND_BUFFER = np.zeros((21, 3))
        RIGHTHAND_BUFFER = np.zeros((21, 3))
        # Empty gesture buffer: [LH, RH]
        LEFTHAND_GESTURE_BUFFER = deque([Gesture(-1, 0) for _ in range(4)])
        RIGHTHAND_GESTURE_BUFFER = deque([Gesture(-1, 0) for _ in range(4)])
        
        # Link main thread to the buffers above
        self.BUFFERS = {
            "LH_LANDMARKS": LEFTHAND_BUFFER,
            "RH_LANDMARKS": RIGHTHAND_BUFFER,
            "LH_GESTURES": LEFTHAND_GESTURE_BUFFER,
            "RH_GESTURES": RIGHTHAND_GESTURE_BUFFER
        }

        # Create an event to trigger the gesture classification FROM the landmark predictor
        EVENT_CLASSIFY = Event()

        self.THREAD_LANDMARK = Thread(target = self.__THREAD_landmark_tracking, args = (EVENT_CLASSIFY, LEFTHAND_BUFFER, RIGHTHAND_BUFFER))
        self.THREAD_LANDMARK.start()

        self.THREAD_GESTURE = Thread(target = self.__THREAD_gesture_classification, args = (EVENT_CLASSIFY, LEFTHAND_GESTURE_BUFFER, RIGHTHAND_GESTURE_BUFFER, LEFTHAND_BUFFER, RIGHTHAND_BUFFER))
        self.THREAD_GESTURE.start()

        while self.COMMS["running"]:
            # Do something
            time.sleep(0.05)
            # Information:
            # print(f"FPS: {self.COMMS['tracking_framerate']}")

            # print("Lefthand in frame: {self.COMMS['LH_in_frame']}")
            # print(f"Righthand in frame: {self.COMMS['RH_in_frame']}")

            # print(f"Lefthand: {', '.join(str((gesture.name, gesture.timestamp)) for gesture in LEFTHAND_GESTURE_BUFFER)}")
            # print(f"Righthand: {', '.join(str((gesture.name, gesture.timestamp)) for gesture in RIGHTHAND_GESTURE_BUFFER)}")
            
            print(f"FPS: {int(self.COMMS['tracking_framerate'])} | LeftHand: {LEFTHAND_GESTURE_BUFFER[0].name} | RightHand: {RIGHTHAND_GESTURE_BUFFER[0].name}" + " "*20, end = "\r")
            sys.stdout.flush()
    def start(self):
        self.THREAD_MAIN = Thread(target = self.__THREAD_main_manager)
        self.THREAD_MAIN.start()

    def stop(self, stopping = True):
        self.COMMS["running"] = False
        if stopping:
            self.THREAD_MAIN.join()
            self.THREAD_GESTURE.join()
            self.THREAD_LANDMARK.join()

    def get_palm_position(self, hand):
        lm_buff = self.BUFFERS["RH_LANDMARKS"] if hand else self.BUFFERS["LH_LANDMARKS"]
        # Subset the palm landmarks
        subset_indices = ()
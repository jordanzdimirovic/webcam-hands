"""
Webcam handtracking / gesture API.
"""
import json
import cv2
import tensorflow as tf
import keras
import os
import vectorfuncs as vect
import numpy as np
from timeit import default_timer as timer
from dataclasses import dataclass
import mediapipe as mp
import time
from collections import deque

from typing import ClassVar

from threading import Thread, Event

import re
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from copy import deepcopy

from math import inf as infinity

# CONSTANTS
DEFAULT_GESTURES = ('open_palm', 'peace', 'fist', 'middle_forwardfacing')
REQUIRED_INPUT_LAYER_SIZE = 64
OPTIONS_WITHOUT_DEFAULTS = ('gesture_names',)

@dataclass
class GestureSnapshot():
    gesture_names: ClassVar[tuple] = tuple()
    id: int
    timestamp: float
    @property
    def name(self):
        return "" if self.id == -1 else self.__class__.gesture_names[self.id]

@dataclass
class TrackingSnapshot():
    data: np.array
    timestamp: float

class WebcamHandsException(Exception):
    pass

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
            "min_tracking_confidence": 0.5,
            "tracking_buffer_length": 5,
            "gesture_buffer_length": 10,
            "hand_velocity_delay": 2,
            "gesture_names": "!REQUIRED!"
        }

    def __perform_checks(self):
        # Assert that hand velocity delay is bounded by the tracking buffer length
        assert 0 < self.options["hand_velocity_delay"] < self.options["tracking_buffer_length"], "Hand velocity delay must be at least one, and bounded by the tracking buffer's size. Check the options"
        # Assert that the model has the correct number of inputs (64)
        input_layer_size = self.GESTURE_MODEL.layers[0].output_shape[1]
        assert input_layer_size == REQUIRED_INPUT_LAYER_SIZE, f"selected model has invalid input layer - expected {REQUIRED_INPUT_LAYER_SIZE}, got {input_layer_size}"

        # Assert that the model has the correct number of outputs
        input_layer_size = self.GESTURE_MODEL.layers[-1].output_shape[1]
        n_gestures = len(self.options["gesture_names"])
        assert input_layer_size == n_gestures, f"selected model has invalid output layer - expected {n_gestures}, got {input_layer_size}"
        
        

    def __init__(self, options = None):
        # Create default options
        if type(options) == str:
            if not os.path.isfile(options):
                # Create it
                open(options, "x").close()
                # Write default options (as a template)
                with open(options, "w") as fileobject:
                    fileobject.write(json.dumps(WebcamHands.get_default_options(), indent = 4))

            with open(options, "r") as jsonf:
                try:
                    options = json.loads(jsonf.read())
                except:
                    raise WebcamHandsException("Option file not a valid JSON file.")

        self.options = type(self).get_default_options()
        self.COMMS = {"running": True, "tracking_framerate": 0, "gestures_pending": [], "LH_in_frame": False, "RH_in_frame": False}
        self.BUFFERS = {
            "LH_LANDMARKS": None,
            "RH_LANDMARKS": None,
            "LH_GESTURES": None,
            "RH_GESTURES": None
        }
        # Ensure that all required options are included
        for x in OPTIONS_WITHOUT_DEFAULTS:
            if x not in options:
                raise WebcamHandsException(f"Unspecified option {x} has no default.")

        # Set all options, provided they're valid
        if options:
            for option in options:
                if option in self.options:
                    self.options[option] = options[option]            
                else:
                    raise ValueError(f"Option '{option}' is not valid.\nFor valid options and their defaults, refer to WebcamHands.get_default_options")
        
        # Set the GestureSnapshot class var
        self.options["gesture_names"] = sorted(self.options["gesture_names"])
        GestureSnapshot.gesture_names = self.options["gesture_names"]

        # Load the gestures model
        model_path = self.options["gesture_classifier_model_path"]
        try:
            self.GESTURE_MODEL = keras.models.load_model(model_path)

        except OSError:
            # Path not found, OR path was not a valid model
            raise WebcamHandsException(f"The gesture classifier model path '{model_path}' was invalid.")

        # Perform checks
        try:
            self.__perform_checks()
        except AssertionError as assertion:
            raise WebcamHandsException(f"API initialisation check failed: {str(assertion)}")

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
                    
                    target_tracking_deque = RIGHTHAND_BUFFER if handedness else LEFTHAND_BUFFER
                    # Rotate
                    
                    target_tracking_deque.rotate()
                    
                    # Get the target buffer to write to
                    target_tracking_snapshot = target_tracking_deque[0]
                    target_tracking_snapshot.timestamp = timer()
                    for i, landmark in enumerate(landmarks.landmark):
                        target_tracking_snapshot.data[i] = [landmark.x, landmark.y, landmark.z]
                    


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

            if self.options["view_camera"]:
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
        # Get the model reference
        gesture_model = self.GESTURE_MODEL
        # Wait for the event to occur
        while self.COMMS["running"]:
            if EVENT_CLASSIFY.wait(0.5):
                pending = self.COMMS["gestures_pending"]
                for hand in pending:
                    selected_tracking_data = RIGHTHAND_BUFFER[0].data if hand else LEFTHAND_BUFFER[0].data
                    # Unnest the buffer data, appending hand to the beginning
                    features = selected_tracking_data.flatten()
                    data_for_mdl = np.array([np.insert(features, 0, hand)])
                    # Predict the gesture
                    gesture_predicted = np.argmax(gesture_model.predict(data_for_mdl)[0])
                    # Push it into the correct deque, pop from the end
                    selected_gesture_deque = RIGHTHAND_GESTURE_BUFFER if hand else LEFTHAND_GESTURE_BUFFER
                    
                    if gesture_predicted != selected_gesture_deque[0].id:
                        selected_gesture_deque.appendleft(
                            GestureSnapshot(
                                id = gesture_predicted,
                                timestamp = timer()
                            )
                        )

                        selected_gesture_deque.pop()

                # Clear the pending list
                self.COMMS["gestures_pending"].clear()
                # Clear the event
                EVENT_CLASSIFY.clear()

    @property
    def velocity(self):
        # Calculate velocity
        return self.calculate_hand_velocity(0, self.options["hand_velocity_delay"]), self.calculate_hand_velocity(1, self.options["hand_velocity_delay"])

    def gesture(self, gest_name):
        gest_names = self.options["gesture_names"]
        if gest_name in gest_names:
            return gest_names.index(gest_name)
        else: raise WebcamHandsException(f"{gest_name} is not a valid gesture name.")

    def __mainthread_loop(self):
        raise NotImplementedError()

    def __THREAD_main_manager(self, event_ready):
        """
            Main thread, created by runtime.
            Responsible for:
            - Managing buffers to facilitate thread communication
            - Create and run the landmark_tracking and gesture_classification threads
        """
        LEFTHAND_BUFFER = deque([TrackingSnapshot(np.zeros((21, 3)), 0) for _ in range(self.options["tracking_buffer_length"])])
        RIGHTHAND_BUFFER = deque([TrackingSnapshot(np.zeros((21, 3)), 0) for _ in range(self.options["tracking_buffer_length"])])
        # Empty gesture buffer: [LH, RH]
        LEFTHAND_GESTURE_BUFFER = deque([GestureSnapshot(-1, 0) for _ in range(self.options["gesture_buffer_length"])])
        RIGHTHAND_GESTURE_BUFFER = deque([GestureSnapshot(-1, 0) for _ in range(self.options["gesture_buffer_length"])])
        
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

        # Tell the runtime thread that we're ready
        event_ready.set()
        # while self.COMMS["running"]:
        #     self.__mainthread_loop()


    def start(self):
        event_mainthread_done = Event()
        self.THREAD_MAIN = Thread(target = self.__THREAD_main_manager, args = (event_mainthread_done,))
        self.THREAD_MAIN.start()
        event_mainthread_done.wait()

    def stop(self, stopping = True):
        self.COMMS["running"] = False
        if stopping:
            self.THREAD_MAIN.join()
            self.THREAD_GESTURE.join()
            self.THREAD_LANDMARK.join()

    def current(self, query):
        # Get the most recent value of a buffer
        if query not in self.BUFFERS:
            raise WebcamHandsException(f"'{query}' is not a valid buffer.")
        return self.BUFFERS[query][0]

    def subset_palm(self, data, moment = 0) -> np.array:
        """
            Obtain the position of the palm
        """
        # Subset the palm landmarks
        subset_indices = [0, 1, 5, 9, 13, 17]
        return data[subset_indices]

    def tracking_snapshots_deltatime(self, hand, ss1: int, ss2: int):
        x = sorted((ss1, ss2))
        buff = self.BUFFERS["RH_LANDMARKS"] if hand else self.BUFFERS["LH_LANDMARKS"]
        ts1, ts2 = buff[x[0]].timestamp, buff[x[1]].timestamp
        if ts1 == 0 or ts2 == 0: return infinity
        return ts1 - ts2

    def calculate_hand_velocity(self, hand, length, normalised = False):
        selected_buff = self.BUFFERS["RH_LANDMARKS"] if hand else self.BUFFERS["LH_LANDMARKS"]
        if length >= len(selected_buff):
            print("Length index out of range! WARNING")
            return np.array([0,0])
        # Get the 2 values required
        latest, first = selected_buff[0], selected_buff[length]
        # Calculate the velocity
        # TODO maybe use palm instead
        vel = vect.difference(vect.average(self.subset_palm(latest.data)), vect.average(self.subset_palm(first.data)))
        # If normalised, divide it by the delta time
        if normalised: vel /= (latest.timestamp - first.timestamp)
        return vel
        


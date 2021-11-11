import tensorflow as tf
import keras
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
import os

gestures = ["open_palm", "claw", "fist", "thumb_up", "peace", "forward_middle_up", "forward_index_up"]

csv_data = []
column_names = ["handLR"] + [f"ordinate{i}" for i in range(63)] + ["gesture"]

for gesture in gestures:
    df = pd.read_csv(os.path.join("data", "gesture_data", gesture, "data.csv"))
    
    csv_data.append()


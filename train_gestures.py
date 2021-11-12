import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Gestures to implement. Note that this isn't used just yet.
gestures = ["open_palm", "claw", "rock_gesture", "fist", "thumb_up", "peace", "forward_middle_up", "forward_index_up"]

csv_data = []
# Define the column names
column_names = ["handLR"] + [f"ordinate{i}" for i in range(63)] + ["gesture"]

for gest in gestures:
    path_to_datafile = os.path.join("data", "gesture_data", gest, "data.csv")
    if os.path.exists(path_to_datafile):
        df = pd.read_csv(path_to_datafile, names = column_names).assign(gesture = gest)
        csv_data.append(df)

df_complete = pd.concat(csv_data)

print(df_complete)

gestures_to_train = list(df_complete.gesture.unique())

df_complete.gesture = df_complete.gesture.map(lambda x: gestures_to_train.index(x))

print("Data for the following gestures has been located:")
for g in df_complete.gesture.unique():
    print(f"Gesture: '{g}: {gestures_to_train[g]}', with {len(df_complete[df_complete.gesture == g])} instances.")

input("Press enter to begin...")

# Perform a train-test-split on the data
X_train, X_test, y_train, y_test = map(tf.convert_to_tensor, train_test_split(df_complete.drop("gesture", axis=1), df_complete.gesture, test_size = 0.2))

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64), # 64 neurons
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(gestures_to_train))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

model.save("gesture_model")
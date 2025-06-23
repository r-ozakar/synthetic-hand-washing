import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

dropout_value = 0.3

base_model = Sequential()
inputs = keras.Input(shape=(96, 96, 3))
base_model.add(inputs)
base_model.add(InceptionV3(include_top=False, pooling='avg'))
base_model.add(Dropout(dropout_value))
base_model.add(Dense(64, activation='relu'))
base_model.add(Dropout(dropout_value))
base_model.add(Dense(8, activation='softmax'))
base_model.load_weights("inception-V3.weights.h5")
base_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

path = "../test data/rgb-roi/"

for file in (os.listdir(path)): 
    img = cv2.imread(path + file)
    # img = cv2.resize(img, (96, 96))
    img = np.reshape(img, (1, 96, 96, -1))
    img = preprocess_input(img)
    prediction = base_model.predict(img)
    print("predicted: " + str(np.argmax(prediction)) + ", actual: " + file[0])
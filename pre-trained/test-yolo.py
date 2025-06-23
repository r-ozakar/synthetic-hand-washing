import os
import cv2
from natsort import realsorted
import numpy as np
import gc
import sys
import numpy as np
from ultralytics import YOLO

base_model = YOLO("./yolo8n.pt")
path = "../test data/rgb/"

for file in (os.listdir(path)):    
    img = cv2.imread(path + file)
    prediction = base_model.predict(img, verbose=False, conf=0.1)
    max = -1
    predicted = -1
    for r in prediction:
        boxes = r.boxes
        for box in boxes:
            c = float(box.conf)
            if (c > max):
                max = c
                predicted = int(box.cls)
    print("predicted: " + str(predicted) + ", actual: " + file[0])
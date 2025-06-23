import os
import cv2
from natsort import realsorted
import numpy as np
import gc
import sys
import numpy as np
from ultralytics import YOLO

base_model = YOLO("./yolo8n-seg.pt")
path = "../test data/rgb/"

for file in (os.listdir(path)):    
    img = cv2.imread(path + file)
    prediction = base_model(img, verbose=False)      
    result = prediction[0]
    masks = result.masks
    predicted = - 1

    if masks is not None:
        for mask in masks:
            polygon = mask.xy[0]
            if len(polygon) >= 3:
                polygon = polygon.astype(np.int64)
                img2 = cv2.drawContours(img, [polygon], -1, (0,255,0), 3)
                cv2.imshow("-", img2)
                cv2.waitKey(1)
        for counter, detection in enumerate(masks.data):
            predicted = int(result.boxes[counter].cls.item())
            print("predicted: " + str(predicted) + ", actual: " + file[0])
            break     
    
    
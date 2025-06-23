import numpy as np
import os
import cv2
import pickle
from natsort import realsorted
from tensorflow.keras.utils import to_categorical

def create_training_data(PATH):
    for char in CHARS:
        for cls in CLASSES:
            cls_num = CLASSES.index(cls)
            for session in SESSION:
                for b in BATHROOM:                   
                    COMBINED_PATH1 = PATH + char + "/" + cls + "/" + session + "/" + b + "/rgb/"
                    COMBINED_PATH2 = PATH + char + "/" + cls + "/" + session + "/" + b + "/segmented/"
                    counter = 0
                    for image in realsorted(os.listdir(COMBINED_PATH1)):
                        print(COMBINED_PATH1 + image)
                        if ".png" in image:
                            img_array = cv2.imread(COMBINED_PATH1 + image)
                            read_segmented = cv2.imread(COMBINED_PATH2 + image)
                            read_segmented = cv2.cvtColor(read_segmented, cv2.COLOR_BGR2GRAY)
                            contours, hierarchy  = cv2.findContours(read_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cnt = contours[0]
                           
                            x,y,w,h = cv2.boundingRect(cnt)
                            voc_bbox = (x, y, x+w, y+h)
                            w = int(w / 2)        
                            midx = x + w
                            boty = y + h + 25
                            topx = midx - 75
                            topy = boty - 150
                            
                            roi = img_array[(topy):(boty), (topx):(topx + 150)]                   
                            roi = cv2.resize(roi, (96, 96))
                            
                            if roi is not None:
                                training_data.append(roi)
                                class_data.append(cls_num)

     
PATH = "../dataset/"

CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8"]
CHARS = ["char1", "char2", "char3", "char4"]
SESSION = ["normal", "soap"]
BATHROOM = ["b1", "b2", "b3", "b4"]

################################################
################### CREATE #####################
################################################ 

training_data = []
class_data = []

create_training_data(PATH)

class_data = to_categorical(class_data, num_classes = 8)
training_data = np.asarray(training_data)

outpath = "./rgb pickle/"
if not os.path.exists(outpath):
    os.makedirs(outpath)
pickle_out = open(outpath + "train_data.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()
pickle_out = open(outpath + "train_classes.pickle", "wb")
pickle.dump(class_data, pickle_out)
pickle_out.close()
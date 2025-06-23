import os
import cv2
import numpy as np
import time
import sys
import pybboxes as pbx

session = ["normal", "soap"]
originpath = "../dataset/"
chars = ["char1", "char2", "char3", "char4"]
training_data = []
class_data = []
W, H = 960, 540

for char in chars:

    current_path = originpath + char + "/"

    for i in range (1, 9):

        for b in range (1, 5):

            for s in session:
                
                current_path = originpath + char + "/" + str(i) + "/" + s + "/b" + str(b)              
                segmented_folder = current_path + "/segmented/"  
                current_path = current_path + "/rgb/"
                
                for img in os.listdir(current_path):

                    if ".png" in img:
                        print(current_path + img)
                        
                        read_img = cv2.imread(current_path + img)
                        read_segmented = cv2.imread(segmented_folder + img)
                        read_segmented = cv2.cvtColor(read_segmented, cv2.COLOR_BGR2GRAY)
                        contours, hierarchy  = cv2.findContours(read_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        cnt = contours[0]
                        x,y,w,h = cv2.boundingRect(cnt)
                        voc_bbox = (x, y, x+w, y+h)
                        bb = pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(W,H))

                        try:
                            file = open(current_path + img[0:-4] + ".txt", "w")
                            file.write(str(i - 1) + " ")                        
                            for k in range(0, len(bb)):
                                file.write(str(bb[k]) + " ")
                            file.close()
                        except Exception as e:
                            print(e)
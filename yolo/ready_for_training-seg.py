import os
import cv2
import numpy as np
import time
import sys
#import shutil
#import pybboxes as pbx

session = ["normal", "soap"]
originpath = "../dataset/"
chars = ["char1", "char2", "char3", "char4"]
W, H = 960, 540

for char in chars:

    current_path = originpath + char + "/"

    for i in range (1, 9):

        for b in range (1, 5):

            for s in session:
                
                current_path = originpath + char + "/" + str(i) + "/" + s + "/b" + str(b)              
                segmented_folder = current_path + "/segmented/"  
                current_path = current_path + "/rgb/"
                
                # for x in os.listdir(current_path):
                    # if ".txt" in x:
                        # os.remove(current_path + x)
                        
                
                for img in os.listdir(current_path):

                    if ".png" in img:
                        print(current_path + img)
                        
                        read_img = cv2.imread(current_path + img)
                        read_segmented = cv2.imread(segmented_folder + img)
                        read_segmented = cv2.cvtColor(read_segmented, cv2.COLOR_BGR2GRAY)
                        contours, hierarchy  = cv2.findContours(read_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        cnt = contours[0]

                        try:
                            if len(cnt) >= 3:
                                file = open(current_path + img[0:-4] + ".txt", "w")
                                file.write(str(i - 1) + " ")
                                for point in cnt:
                                    file.write(str(point[0][0]/W) + " ")
                                    file.write(str(point[0][1]/H) + " ")
                                file.close()
                        except Exception as e:
                            print(e)

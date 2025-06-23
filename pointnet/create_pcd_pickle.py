import os
import cv2
import pickle
import numpy as np
import open3d as o3d
from natsort import realsorted
from keras.utils import to_categorical
from sklearn.preprocessing import normalize

def create_training_data(PATH):
    
    for char in CHARS:
        
        for cls in CLASSES:
            
            cls_num = CLASSES.index(cls)
            
            for session in SESSION:
                
                for b in BATHROOM:    
                    
                    DEPTH_PATH = PATH + char + "/" + cls + "/" + session + "/" + b + "/depth isolated/"
                    SEGMENTED_PATH = PATH + char + "/" + cls + "/" + session + "/" + b + "/segmented/"

                    for file in realsorted(os.listdir(DEPTH_PATH)):
                        
                        if ".png" in file:
                            
                            print(DEPTH_PATH + file)
                            print(SEGMENTED_PATH + file[5:])

                            read_depth = cv2.imread(DEPTH_PATH + file, -1)
                            read_segmented = cv2.imread(SEGMENTED_PATH + file[5:], -1)
                            contours, hierarchy  = cv2.findContours(read_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            cnt = contours[0]
                            x,y,w,h = cv2.boundingRect(cnt)
                            w = int(w / 2)        
                            midx = x + w
                            boty = y + h + 25
                            topx = midx - 75
                            topy = boty - 150

                            depth_roi = read_depth[(topy):(boty), (topx):(topx + 150)]                   
                            depth_roi = cv2.resize(depth_roi, (150, 150))
                            depth_roi = np.asarray(depth_roi)

                            lowestj = 0
                            lowesti = 0
                            value = 0
                            z = 0
                            
                            for i in range (0, depth_roi.shape[0]): #height
                                for j in range (0, depth_roi.shape[1]): #width
                                    if depth_roi[i][j] > 100:
                                        value = depth_roi[i][j] #find tip point's z value
                                        
                            z = value / 1000
                            depth_roi = o3d.geometry.Image(depth_roi)
                            
                            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_roi,
                                                            intrinsic,
                                                            np.identity(4),
                                                            depth_scale=1000.0,
                                                            depth_trunc=1000.0)

                            pcd.transform([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

                            sample = np.asarray(pcd.points)

                            sample2 = []
                            sample2.clear()

                            for i in range (0, sample.shape[0]): 
                                if sample[i][2] > z - 1 and sample[i][2] < z + 3 and sample[i][1] > -1.8:
                                    sample2.append(sample[i])
                            
                            sample2 = np.asarray(sample2)

                            if sample2 is not None and len(sample2) >= 512:
                                pcd2 = o3d.geometry.PointCloud()
                                pcd2.points = o3d.utility.Vector3dVector(sample2)
                                center = pcd2.get_center()
                                pcd2.translate(-center)

                                # vis.clear_geometries()
                                # vis.add_geometry(pcd2)
                                # vis.poll_events()
                                # vis.update_renderer()
                                
                                # o3d.visualization.draw_geometries([pcd2])
                                # o3d.io.write_point_cloud()

                                sample3 = np.asarray(pcd2.points)
                                sample3 = normalize(sample3, norm="l1")
                                sample3 = sample3[np.random.choice(np.arange(len(sample3)), 512)]

                                training_data.append(sample3)
                                class_data.append(cls_num)

width=960 
height=540 
ppx=480 
ppy=270 
fx=7000
fy=7000
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)
                    
PATH = "../dataset/"

CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8"]
CHARS = ["char1", "char2", "char3", "char4"]
SESSION = ["normal", "soap"]
BATHROOM = ["b1", "b2", "b3", "b4"]

################################################
################### CREATE #####################
################################################ 

# vis = o3d.visualization.Visualizer() 
# vis.create_window()

training_data = []
class_data = []

create_training_data(PATH)
class_data = to_categorical(class_data, num_classes = 8)

outpath = "./pcd pickle/"
if not os.path.exists(outpath):
    os.makedirs(outpath)
pickle_out = open(outpath + "train_data.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()
pickle_out = open(outpath + "train_classes.pickle", "wb")
pickle.dump(class_data, pickle_out)
pickle_out.close()
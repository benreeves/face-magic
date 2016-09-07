import csv
import numpy as np
import pandas as pd
import cv2
import os

f = "adienData/all_faces.csv"
path = "adienData/faces_cropped/"
i=0
j=0
fname = 'face_labels_'
data_set = []
labels = []
reader=csv.reader(open(f,"rt"),delimiter=",")
with open('flattened_faces.csv', 'w', newline='') as csvfile:
    for row in reader:
        folder = row[0] 
        img = "coarse_tilt_aligned_face." + row[2] + "." + row[1]
        ff = path+folder+"/"+img
        if(row[3] == "None"):
            print('Skipping record')
            continue
        if(os.path.isfile(ff)):
            #img = cv2.imread(ff,0)
            #x = img.flatten()
            if (i<1000):
                x = [row[4], row[3]]
                labels.append(x)
                #data_set.append(x)
            else:
                #mat = np.array(data_set)
                mat = np.array(labels)
                np.save(fname+str(j)+'.npy', mat)
                i = -1
                j += 1
                #data_set = []
                labels = []
            i += 1


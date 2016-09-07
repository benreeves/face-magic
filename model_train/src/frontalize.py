import dlib
import os
import sys
import frontalization.frontalize as frontalize
import frontalization.facial_feature_detector as feature_detection
import frontalization.camera_calibration as calib
import cv2
import numpy as np
import scipy.io as io
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    face_detector = dlib.get_frontal_face_detector();
    this_path = os.path.dirname(os.path.abspath(__file__))

    for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
        for f in [ff for ff in filenames if ff.endswith(".jpg")]:
            print("processing file: {}".format(f))
            im = dirpath + "/" + f
            img = cv2.imread(im)
            dets = face_detector(img)
            #remove data with two faces since we don't have well defined metadata in that case
            if len(dets) > 1 or len(dets) == 0:
                continue

            print("{} faces detected".format(len(dets)))
            lmarks = feature_detection.get_landmarks(img)
            model3D = frontalize.ThreeD_Model(this_path +"/frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib')
            #iioioo perform camera calibration according to the first face detected
            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
            # load mask to exclude eyes from symmetry
            eyemask = np.asarray(io.loadmat('frontalization/frontalization_models/eyemask.mat')['eyemask'])
            # perform frontalization
            frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
            frontal_face = frontal_sym[:, :, ::-1]
            print("frontal_face shape: {}".format(frontal_face.shape))
            #at this point, we don't know if the frontalization fucked out faces into grotesque monsters. Let's see if 
            #we can recognize a face in there, and if not we'll just skip this record
            temp = Image.fromarray(frontal_face)
            temp.save(im[:-3]+"_temp.jpg")
            img = cv2.imread(im[:-3]+"_temp.jpg")

            dets = face_detector(img)
            if len(dets) != 1:
                print("It seems the fronatlization fucked {} and dlib can't find the face any more".format(f)) 
                continue
            img = Image.fromarray(img)
            img.save(im)
            os.remove(im[:-3]+"_temp.jpg")


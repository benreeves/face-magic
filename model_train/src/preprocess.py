import dlib
import os
import sys
import frontalization.frontalize as frontalize
import frontalization.facial_feature_detector as feature_detection
import frontalization.camera_calibration as calib
import cv2
import numpy as np
import frontalization.check_resources as check
import scipy.io as io
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    face_detector = dlib.get_frontal_face_detector();
    this_path = os.path.dirname(os.path.abspath(__file__))

    for f in sys.argv[1:]:
        print("processing file: {}".format(f))
        win = dlib.image_window()
        #img = Image.open(f).convert('L')
        #img.save(f)
        #img.save('test_gray.jpg')
        print('Processing')
        img = cv2.imread(f,0)
        dets = face_detector(img)
        #remove data with two faces since we don't have well defined metadata in that case
        if len(dets) > 1:
            os.remove(f)
            continue
        for i, d in enumerate(dets):
            print("Detection {}: Left: {}, Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
            x_start = d.left()
            y_start = d.top()
            w = d.right() - d.left()
            h = d.bottom() - d.top()
            xtra_h = int(0.2*h)
            xtra_w = int(0.2*w)
            new_image = img[y_start-xtra_h:y_start+h+xtra_h, x_start-xtra_w:x_start+w+xtra_w].copy()
            new_image = cv2.resize(new_image, (250, 250))
            im = Image.fromarray(new_image)
            im.save(f)

        img = cv2.imread(f)
        lmarks = feature_detection.get_landmarks(img)
        model3D = frontalize.ThreeD_Model(this_path +"/frontalization/frontalization_models/model3Ddlib.mat", 'model_dlib')
        plt.figure()
        plt.title('Landmarks Detected')
        plt.imshow(img[:, :, ::-1])
        plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])
        #iioioo perform camera calibration according to the first face detected
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
        # load mask to exclude eyes from symmetry
        eyemask = np.asarray(io.loadmat('frontalization/frontalization_models/eyemask.mat')['eyemask'])
        # perform frontalization
        frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
        plt.figure()
        plt.title('Frontalized with soft symmetry')
        plt.imshow(frontal_sym[:, :, ::-1])
        plt.show()
        dlib.hit_enter_to_continue()


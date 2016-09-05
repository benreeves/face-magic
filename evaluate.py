import dlib
from scipy import ndimage
from scipy import misc
import numpy as np
import keras_model

def image_to_numpy(img):

    face = misc.imread(img, mode='L')
    face_detector = dlib.get_frontal_face_detector();
    results = []

    dets = face_detector(face)
    if len(dets) == 0:
        return "Dlib couldn't detect a face... choose another image!"
    for i, d in enumerate(dets):
        x_start = d.left()
        y_start = d.top()
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        xtra_h = int(0.*h)
        xtra_w = int(0.*w)
        new_image = face[y_start-xtra_h:y_start+h+xtra_h, x_start-xtra_w:x_start+w+xtra_w].copy()
        new_image = misc.imresize(new_image, (112, 112))
        results.append(new_image)

    return results


import dlib
import os
import sys
import cv2
import numpy as np
import scipy.io as io
from PIL import Image
import matplotlib.pyplot as plt
import pdb

def normalize(arr):
    """
    Linear brightness normalizer
    """
    arr=arr.astype('float')
    minval = arr.min()
    maxval = arr.max()
    if minval != maxval:
        arr -= minval
        arr *= (255.0/(maxval-minval))

    return arr.astype('uint8')

def crop(img, frame):
    
    m, n = img.shape
    bottom, top, left, right = 0, m, 0, n
    x_start = frame.left()
    y_start = frame.top()
    w = frame.right() - frame.left()
    h = frame.bottom() - frame.top()
    xtra_h = int(0.2*h)
    xtra_w = int(0.0*w)
    if y_start-xtra_h > 0:
        bottom = y_start-xtra_h
    if y_start+h+xtra_h < m:
        top = y_start+h+xtra_h
    if x_start-xtra_w > 0:
        left = x_start-xtra_w
    if x_start+w+xtra_w < n:
        right = x_start+w+xtra_w

#    pdb.set_trace();

    return img[bottom:top, left:right].copy()

if __name__ == '__main__':

    face_detector = dlib.get_frontal_face_detector();
    this_path = os.path.dirname(os.path.abspath(__file__))
    
    for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
        for f in [ff for ff in filenames if ff.endswith(".jpg")]:
            print("processing file: {}".format(f))
            image_path = dirpath + "/" + f
            print(image_path)
            img = cv2.imread(image_path,0)
            dets = face_detector(img)
            #remove data with two faces since we don't have well defined metadata in that case
            if (len(dets) > 1) or (len(dets) == 0):
                if(os.path.isfile(image_path)):
                    os.remove(image_path)
                if(os.path.isfile(image_path[:-3]+"txt")):
                    os.remove(image_path[:-3]+"txt")
                continue

            #Really, no need to be enumerating but fuck it right
            for i, d in enumerate(dets):
                print("Detection {}: Left: {}, Top: {} Right: {} Bottom: {}".format(
                    i, d.left(), d.top(), d.right(), d.bottom()))

                #x_start = d.left()
                #y_start = d.top()
                #w = d.right() - d.left()
                #h = d.bottom() - d.top()
                #xtra_h = int(0.2*h)
                #xtra_w = int(0.0*w)
                ##TODO add a check to make sure we don't crop more than what is available
                #new_image = img[y_start-xtra_h:y_start+h+xtra_h, x_start-xtra_w:x_start+w+xtra_w].copy()
                new_image = crop(img, d)
                try:
                    new_image = cv2.resize(new_image, (200, 280))
                    norm = normalize(new_image)
                    im = Image.fromarray(new_image)
                    im.save(image_path)
                except:
                    if(os.path.isfile(image_path)):
                        os.remove(image_path)
                    if(os.path.isfile(image_path[:-3]+"txt")):
                        os.remove(image_path[:-3]+"txt")
                    continue

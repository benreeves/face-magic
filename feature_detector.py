import dlib
import uuid
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FeatureDetector:
    """An object for detecting and overlaying facial features to make things look cool"""

    def __init__(self, image_path, landmark_path):
        self.facial_feature_path = landmark_path
        self.ShapePredictor = dlib.shape_predictor(landmark_path)
        self.Detector = dlib.get_frontal_face_detector()
        self._img_path = image_path

    def _shape_to_np(self, shape):
        xy = []
        for i in range(68):
            xy.append((shape.part(i).x, shape.part(i).y,))
        xy = np.asarray(xy, dtype='float32')
        return xy


    def save_landmarks(self, img):
        """Draws the face detection box and facial landmarks to
        a plot and returns the location of the saved plot"""
        faces = misc.imread(img, mode='L')
        lmarks = []
        border_box = []
        dets = self.Detector(faces)
        shapes = []
        for k, det in enumerate(dets):
            left, right = det.left(), det.right()
            top, bottom = det.top(), det.bottom()
            shape = self.ShapePredictor(faces, det)
            shapes.append(shape)
            xy = self._shape_to_np(shape)
            lmarks.append(xy)
            border_box.append((left, right, bottom, top))

        lmarks = np.asarray(lmarks, dtype='float32')
        return self._draw_landmarks(faces, border_box, lmarks)
        
    def _draw_landmarks(self, face, border_box, lmarks):
        #TODO handle multiple faces in an image
        fig = plt.figure()
        plt.title('Feature Landmarks')
        plt.imshow(face, cmap='gray')
        plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])
        currentAxis = plt.gca() #get current axis for overlay
        width = border_box[0][1] - border_box[0][0]
        height = border_box[0][2] - border_box[0][3]
        currentAxis.add_patch(
                patches.Rectangle(
                    (border_box[0][0], border_box[0][3]),
                    width, height,
                    fill = False, edgecolor='red')
                )
        filename = str(uuid.uuid4()) + '.png'
        file_path = os.path.join(self._img_path, filename)
        fig.savefig(file_path)
        return filename


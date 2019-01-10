import numpy as np
import cv2
import imutils

from imutils import face_utils


class Face:
    def __init__(self, shape, frame, dim):
        self.shape = shape
        self.frame = frame
        self.dim = dim

    def release(self):
        del self.shape
        del self.frame
        del self.dim

    def extract_facial_landmarks(self, name):
        facial_landmarks = {"column": [], "value": []}

        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[name]
        for facial_i in range(i, j):
            tmp = np.array([self.shape[facial_i]])
            facial_landmarks["value"].append(tmp[0][0])
            facial_landmarks["column"].append(str(name + "." + str(facial_i) + ".x"))
            facial_landmarks["value"].append(tmp[0][1])
            facial_landmarks["column"].append(str(name + "." + str(facial_i) + ".y"))

        return facial_landmarks

    def extract_roi(self, name):
        (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[name]

        (x, y, w, h) = cv2.boundingRect(np.array([self.shape[i:j]]))
        roi = self.frame.image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        return roi

    def generate_shape_on_frame(self):
        return face_utils.visualize_facial_landmarks(self.frame.image, self.shape)

    def generate_shape(self):
        blank_image = np.zeros(self.frame.image.shape, np.uint8)
        return face_utils.visualize_facial_landmarks(blank_image, self.shape)

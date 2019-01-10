import cv2
import imutils
import dlib
from local_library.Face import Face

from imutils import face_utils


# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("local_library/shape_predictor_68_face_landmarks.dat")
FACEPARTSNAME = ["jaw", "left_eyebrow", "nose", "mouth", "right_eyebrow", "left_eye", "right_eye"]


class Frame:
    def __init__(self, nframe, image):
        self.nframe = nframe
        self.image = image
        self.faces = []

    def extract_shape_faces(self):
        image = imutils.resize(self.image, width=500)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then convert the landmark (x, y)-coordinates
            # to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            self.faces.append(Face(shape, self, image.shape))

    def release(self):
        del self.nframe
        del self.image
        for face in self.faces:
            face.release()
        del self.faces

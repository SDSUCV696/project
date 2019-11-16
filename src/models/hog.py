import dlib
from imutils import face_utils


class Hog:
    def __init__(self):
        self.face_detect = dlib.get_frontal_face_detector()

    def detect_face(self, gray):
        rects = self.face_detect(gray, 1)
        return rects

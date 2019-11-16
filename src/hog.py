import dlib
from imutils import face_utils


class Hog:
    def detect_face(gray):
        face_detect = dlib.get_frontal_face_detector()
        rects = face_detect(gray, 1)
        return rects

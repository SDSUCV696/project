import dlib


class Cnn:
    def __init__(self):
        path = "/home/jahn/Documents/CV696/project/utils/mmod_human_face_detector.dat"
        self.dnnFaceDetector = dlib.cnn_face_detection_model_v1(path)

    def detect_face(self, gray):
        rects = self.dnnFaceDetector(gray, 0)
        return rects


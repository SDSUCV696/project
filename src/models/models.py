import cv2
import dlib
from imutils import face_utils


class Model:
    def __init__(self, file):
        self.file = open(file, "w")

    def detect_face(self):
        pass

    def convert(self):
        pass

    def write(self, string):
        self.file.write(string + "\n")

    def close(self):
        self.file.close()


class Hog(Model):
    def __init__(self):
        super(Hog, self).__init__("hog_output.txt")
        self.face_detect = dlib.get_frontal_face_detector()

    def detect_face(self, gray):
        rects = self.face_detect(gray, 1)
        return rects

    def convert(self, rects):
        faces = []
        for (i, rect) in enumerate(rects):
            faces.append(face_utils.rect_to_bb(rect))
            (x, y, w, h) = faces[-1]
        return faces


class Cnn(Model):
    def __init__(self):
        super(Cnn, self).__init__("cnn_output.txt")
        path = "/home/jahn/Documents/CV696/project/utils/mmod_human_face_detector.dat"
        self.dnnFaceDetector = dlib.cnn_face_detection_model_v1(path)

    def detect_face(self, gray):
        rects = self.dnnFaceDetector(gray, 0)
        return rects

    def convert(self, rects):
        faces = []
        for (i, rect) in enumerate(rects):
            x1 = rect.rect.left()
            y1 = rect.rect.top()
            x2 = rect.rect.right()
            y2 = rect.rect.bottom()
            faces.append([x1, y1, x2 - x1, y2 - y1])
            # Rectangle around the face
        return faces


class Cascade(Model):
    def __init__(self):
        super(Cascade, self).__init__("cascade_output.txt")
        casc_path = "../utils/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(casc_path)

    def detect_face(self, gray):
        # Detect faces
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def convert(self, rects):
        return rects

    def write(self, string):
        self.file.write(string + "\n")

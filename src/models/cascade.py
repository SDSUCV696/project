import cv2


class Cascade:
    def __init__(self):
        cascPath = "../utils/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect_face(self, gray):
        # Detect faces
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces


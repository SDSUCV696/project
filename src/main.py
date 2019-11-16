import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from models.cascade import Cascade
from models.hog import Hog
from models.cnn import Cnn


def main():
    gray = cv2.imread('../test/img2.jpg', 0)
    """
    model1 = Cascade()
    faces = model1.detect_face(gray)
    for (x, y, w, h) in faces: 
        # Draw rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

    model2 = Hog()
    rects = model2.detect_face(gray)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)

    """
    model3 = Cnn()
    rects = model3.detect_face(gray)
    for (i, rect) in enumerate(rects):
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
	# Rectangle around the face
        cv2.rectangle(gray, (x1, y1), (x2, y2), (255, 255, 255), 3)

    plt.figure(figsize=(12,8))
    plt.imshow(gray, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

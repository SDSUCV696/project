import cv2
import matplotlib.pyplot as plt
import numpy as np
from models.cascade import Cascade
from models.hog import Hog
from imutils import face_utils


def main():
    gray = cv2.imread('../test_img/img2.jpg', 0)
    """
    faces = Cascade.detect_face(gray)
    for (x, y, w, h) in faces: 
        # Draw rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)
    """

    rects = Hog.detect_face(gray)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)
        
    plt.figure(figsize=(12,8))
    plt.imshow(gray, cmap='gray')
    plt.show()     

if __name__ == "__main__":
    main()

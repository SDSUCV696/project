import matplotlib.pyplot as plt
import numpy as np
import dlib
from imutils import face_utils
from cascade import *


def main():
    gray = cv2.imread('img2.jpg', 0)
    faces = detect_face(gray)
    # For each face
    for (x, y, w, h) in faces: 
        # Draw rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

    plt.figure(figsize=(12,8))
    plt.imshow(gray, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()

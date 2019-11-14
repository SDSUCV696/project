import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

cascPath = "/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

gray = cv2.imread('img.jpg', 0)

plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()

# Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    flags=cv2.CASCADE_SCALE_IMAGE
)
    
# For each face
for (x, y, w, h) in faces: 
    # Draw rectangle around the face
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()

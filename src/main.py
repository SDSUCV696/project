import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from models.cascade import Cascade
from models.hog import Hog
from models.cnn import Cnn

def parse_input():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = "../test/wider_face_split/wider_face_train_bbx_gt.txt"
    gt_bbxs = {}
    gt_file = open(os.path.join(script_dir, gt_path))
    line = gt_file.readline().replace('\n', '')
    while line:
        gt_bbxs[line] = []
        num_bbxs = gt_file.readline()
        if int(num_bbxs) !=  0:
            for i in range(int(num_bbxs)):
                bbx = gt_file.readline()
                bbx = bbx.replace('\n', '')
                tokens = bbx.split(' ')
                tokens.pop() # last char is empty from '\n'
                int_tokens = [int(i) for i in tokens]
                gt_bbxs[line].append(int_tokens)
        else:
            line = gt_file.readline().replace('\n', '')
        line = gt_file.readline().replace('\n', '')
        if not line:
            break
    return gt_bbxs


def main():
    gray = cv2.imread('../test/img3.jpg', 0)
    # dict {'img_name' : [ bbx1, bbx2], ... } where each bbx is an array (4 + 6 tuple)
    gt_bbxs = parse_input()

    """
    print(fp.readline())
    n = fp.readline()
    gt_rects = []
    for i in n:
        gt_rects.append(fp.read())
    print(gt_rects[0])
    """

    """
    model1 = Cascade()
    faces = model1.detect_face(gray)
    for (x, y, w, h) in faces: 
        # Draw rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

    plt.figure(figsize=(12,8))
    plt.imshow(gray, cmap='gray')
    plt.show()

    model2 = Hog()
    rects = model2.detect_face(gray)
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)

    plt.figure(figsize=(12,8))
    plt.imshow(gray, cmap='gray')
    plt.show()
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
    """


if __name__ == "__main__":
    main()

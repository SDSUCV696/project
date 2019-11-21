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

def overlap(rect_A, rect_B):
    A = [ [rect_A[0], rect_A[1]], [rect_A[0] + rect_A[2], rect_A[1] + rect_A[3]] ]
    B = [ [rect_B[0], rect_B[1]], [rect_B[0] + rect_B[2], rect_B[1] + rect_B[3]] ]

    if A[0][0] > B[1][0] or A[1][0] < B[0][0]:
        return False

    if A[1][1] < B[0][1] or A[0][1] > B[1][1]:
        return False

    return True

def IoU(rect_A, rect_B):
    Ax1 = rect_A[0]
    Ay1 = rect_A[1]
    Ax2 = Ax1 + rect_A[2]
    Ay2 = Ay1 + rect_A[3]

    Bx1 = rect_B[0]
    By1 = rect_B[1]
    Bx2 = Bx1 + rect_B[2]
    By2 = By1 + rect_B[3]

    x1 = max(Ax1, Bx1)
    y1 = max(Ay1, By1)
    x2 = min(Ax2, Bx2)
    y2 = min(Ay2, By2)

    intersection = abs(x2 - x1) * abs(y2 - y1)
    union = abs(Ax2 - Ax1) * abs(Ay2 - Ay1) + abs(Bx2 - Bx1) * abs(By2 - By1) - intersection
    iou = intersection / union
    return iou

def main():
    gray = cv2.imread('../test/img3.jpg', 0)
    # dict {'img_name' : [ bbx1, bbx2], ... } where each bbx is an array (4 + 6 tuple)
    # gt_bbxs = parse_input()

    
    print(IoU([1, 1, 3, 2], [5, 1, 1, 1]))

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

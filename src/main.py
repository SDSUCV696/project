import os
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
from src.models.cascade import Cascade
from src.models.hog import Hog
from src.models.cnn import Cnn


THRESHOLD = 0.5


def parse_input():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = "../test/wider_face_split/wider_face_val_bbx_gt.txt"
    gt_bbxs = {}
    gt_file = open(os.path.join(script_dir, gt_path))
    line = gt_file.readline().replace('\n', '').split("/")[1]
    while line:
        gt_bbxs[line] = []
        num_bbxs = gt_file.readline()
        if int(num_bbxs) != 0:
            for i in range(int(num_bbxs)):
                bbx = gt_file.readline()
                bbx = bbx.replace('\n', '')
                tokens = bbx.split(' ')
                # last char is empty from '\n'
                tokens.pop()
                int_tokens = [int(i) for i in tokens]
                gt_bbxs[line].append(int_tokens)
        else:
            # if the num of boxes is zero, move line down by one
            line = gt_file.readline()
        line = gt_file.readline().replace('\n', '')
        if line:
            # if line is not EOF, split it
            line = line.split("/")[1]
        else:
            break
    return gt_bbxs


def overlap(rect_a, rect_b):
    a = [[rect_a[0], rect_a[1]], [rect_a[0] + rect_a[2], rect_a[1] + rect_a[3]]]
    b = [[rect_b[0], rect_b[1]], [rect_b[0] + rect_b[2], rect_b[1] + rect_b[3]]]

    if a[0][0] > b[1][0] or a[1][0] < b[0][0]:
        return False

    if a[1][1] < b[0][1] or a[0][1] > b[1][1]:
        return False

    return True


def iou(rect_a, rect_b):
    ax1 = rect_a[0]
    ay1 = rect_a[1]
    ax2 = ax1 + rect_a[2]
    ay2 = ay1 + rect_a[3]

    bx1 = rect_b[0]
    by1 = rect_b[1]
    bx2 = bx1 + rect_b[2]
    by2 = by1 + rect_b[3]

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = abs(x2 - x1) * abs(y2 - y1)
    union = abs(ax2 - ax1) * abs(ay2 - ay1) + abs(bx2 - bx1) * abs(by2 - by1) - intersection
    return intersection / union


def compare_exp_to_gt(exp_bbxs, gt_bbxs, threshold):
    """ args:
    exp_bbx - a list of bbx's produced by the models
    gt_bbx - a set of gt bbx's for the image { [....], ... , [....] }

    where bbx = [x, y, w, h]

    return: [ true_pos, false_pos, false_neg ]
    """
    true_pos = 0
    for bbx in exp_bbxs:
        found = False
        match = []
        for gt in gt_bbxs:
            if overlap(bbx, gt[0:4]) and iou(bbx, gt) >= threshold:
                found = True
                match = gt
                break
        if found:
            true_pos = true_pos + 1
            gt_bbxs.remove(match)
            if len(gt_bbxs) == 0:
                break

    false_pos = len(exp_bbxs) - true_pos
    false_neg = len(gt_bbxs)
    return [true_pos, false_pos, false_neg]


def main():
    gray = cv2.imread('../test/7_Cheering_Cheering_7_125.jpg', 0)
    # dict {'img_name' : [ [bbx1], [bbx2] ], 'img2' : [ ]... } where each bbx is an array (4 + 6 tuple)
    all_gt_bbxs = parse_input()
    gt_bbxs = all_gt_bbxs['7_Cheering_Cheering_7_125.jpg']

    """
    model1 = Cascade()
    faces = model1.detect_face(gray)
    for (x, y, w, h) in faces: 
        # Draw rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

    plt.figure(figsize=(12, 8))
    plt.imshow(gray, cmap='gray')
    plt.show()
    data = compare_exp_to_gt(faces, gt_bbxs, THRESHOLD)
    print(data)
    """

    """
    model2 = Hog()
    rects = model2.detect_face(gray)
    faces = []
    for (i, rect) in enumerate(rects):
        faces.append(face_utils.rect_to_bb(rect))
        (x, y, w, h) = faces[-1]
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)

    plt.figure(figsize=(12, 8))
    plt.imshow(gray, cmap='gray')
    plt.show()
    """

    model3 = Cnn()
    rects = model3.detect_face(gray)
    faces = []
    for (i, rect) in enumerate(rects):
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
        faces.append([x1, y1, x2-x1, y2-y1])
        # Rectangle around the face
        cv2.rectangle(gray, (x1, y1), (x2, y2), (255, 255, 255), 3)

    plt.figure(figsize=(12, 8))
    plt.imshow(gray, cmap='gray')
    plt.show()
    data = compare_exp_to_gt(faces, gt_bbxs, THRESHOLD)
    print(data)


if __name__ == "__main__":
    main()

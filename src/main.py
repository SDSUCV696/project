import os
import cv2
import numpy as np
import time
from src.models import models


THRESHOLD = 0.5
ALL_GT_BBXS = {}
MAX_BLUR = 1
MAX_OCC = 1
MIN_AREA = 35*35
# [blur, expression, illum, invalid(?), occlusion, pose]
TOTAL_ATTRIBUTES = [0 for i in range(6)]


def collect_gt_values():
    global TOTAL_ATTRIBUTES
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = "../test/wider_face_split/wider_face_val_bbx_gt.txt"
    gt_file = open(os.path.join(script_dir, gt_path))
    line = gt_file.readline().replace('\n', '').split("/")[1]
    while line:
        ALL_GT_BBXS[line] = []
        num_bbxs = gt_file.readline()
        if int(num_bbxs) != 0:
            for i in range(int(num_bbxs)):
                bbx = gt_file.readline()
                bbx = bbx.replace('\n', '')
                tokens = bbx.split(' ')
                # last char is empty from '\n'
                tokens.pop()
                gt = [int(i) for i in tokens]

                # omit bbxs that have too much blur, occlusion, are invalid, or too little area
                if gt[2] * gt[3] < MIN_AREA or gt[4] > MAX_BLUR or gt[8] > MAX_OCC or gt[9] == 1:
                    continue
                else:
                    # 'img_name' -> [0,1,...,9]
                    ALL_GT_BBXS[line].append(gt)
                    TOTAL_ATTRIBUTES = np.add(TOTAL_ATTRIBUTES, gt[4:])
            # remove images with 0 gt bbxs
            if len(ALL_GT_BBXS[line]) == 0:
                del ALL_GT_BBXS[line]
        else:
            # if the num of boxes is zero, manually move line down by one
            line = gt_file.readline()
        line = gt_file.readline().replace('\n', '')
        if line:
            # if line is not EOF, split it
            line = line.split("/")[1]
        else:
            break


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
    true_pos_attributes = [0 for i in range(6)]
    false_neg_attributes = [0 for i in range(6)]
    for bbx in exp_bbxs:
        found = False
        gt_match = []
        for gt in gt_bbxs:
            if overlap(bbx, gt[0:4]) and iou(bbx, gt) >= threshold:
                found = True
                gt_match = gt
                break
        if found:
            true_pos = true_pos + 1
            true_pos_attributes = np.add(gt_match[4:], true_pos_attributes)
            gt_bbxs.remove(gt_match)
            if len(gt_bbxs) == 0:
                break

    # number of incorrect bbxs
    false_pos = len(exp_bbxs) - true_pos
    false_neg = len(gt_bbxs)

    # number of correct bbxs remaining (unmatched)
    for gt in gt_bbxs:
        false_neg_attributes = np.add(false_neg_attributes, gt[4:])

    return [true_pos, false_pos, false_neg], true_pos_attributes, false_neg_attributes


def go(model):
    total_confusion_matrix = [0, 0, 0]
    total_true_pos_attributes = [0 for i in range(6)]
    total_false_neg_attributes = [0 for i in range(6)]
    total_number_of_gt_bbxs = 0
    i = 0
    sz = len(ALL_GT_BBXS)
    t_start = time.clock()

    for img, gt_bbxs in ALL_GT_BBXS.items():
        number_of_correct_gt_bbxs = len(gt_bbxs)
        i = i + 1
        total_number_of_gt_bbxs = total_number_of_gt_bbxs + number_of_correct_gt_bbxs
        gray = cv2.imread("../test/WIDER_val_images/" + img)
        rects = model.detect_face(gray)
        faces = model.convert(rects)
        confusion_matrix, true_pos_attributes, false_neg_attributes \
            = compare_exp_to_gt(faces, gt_bbxs, THRESHOLD)
        total_confusion_matrix = np.add(total_confusion_matrix, confusion_matrix)
        total_true_pos_attributes = np.add(total_true_pos_attributes, true_pos_attributes)
        total_false_neg_attributes = np.add(total_false_neg_attributes, false_neg_attributes)
        string = "{0}/{1} true_positive_ratio: {2}/{3} total_percent_correct: {4:0.9f}"\
                 " {5}".format\
            (i, sz, confusion_matrix[0], number_of_correct_gt_bbxs,
             (total_confusion_matrix[0]/total_number_of_gt_bbxs*100), img)
        model.write(string)
        print(string)

    true_positive_ratios = []
    for i, val in enumerate(TOTAL_ATTRIBUTES):
        attribute_string = (str(total_true_pos_attributes[i])) + "/" + (str(val))
        true_positive_ratios.append(attribute_string)

    string = "true  positives: blur:{0}, expr:{1}, illum:{2}, occlu:{3}".format\
        (true_positive_ratios[0], true_positive_ratios[1], true_positive_ratios[2], true_positive_ratios[4])
    model.write(string)

    string = "false negatives: blur:{0}, expr:{1}, illum:{2}, occlu:{3}".format\
        (total_false_neg_attributes[0], total_false_neg_attributes[1], total_false_neg_attributes[2],
         total_false_neg_attributes[4])
    model.write(string)

    t_end = time.clock()
    time_elapsed = "total time elapsed:{0:5.2f}".format(t_end - t_start)
    model.write(time_elapsed)
    model.close()
    print(time_elapsed)
    return total_confusion_matrix


def main():
    # dict {'img_name' : [ [bbx1], [bbx2] ], 'img2' : [ ]... } where each bbx is an array (4 + 6 tuple)
    collect_gt_values()

    hog = models.Hog()
    go(hog)

    """
    cascade = models.Cascade()
    go(cascade)
    
    
    cnn = models.Cnn()
    go(cnn)
    """


if __name__ == "__main__":
    main()

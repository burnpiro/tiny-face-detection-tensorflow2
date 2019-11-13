import cv2
import numpy as np


def draw_outputs(img, boxes, draw_labels=True):
    for y_c, x_c, h, w, _ in boxes:
        x0 = img.shape[1] * (x_c - w / 2)
        y0 = img.shape[0] * (y_c - h / 2)
        x1 = x0 + img.shape[1] * w
        y1 = y0 + img.shape[0] * h
        x1y1 = tuple(np.array([x0, y0]).astype(np.int32))
        x2y2 = tuple(np.array([x1, y1]).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        if draw_labels:
            img = cv2.putText(img, '{} {:.4f}'.format(
                'human', w),
                              x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

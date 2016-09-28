from __future__ import print_function
from sys import argv
import os.path

import cv2
import numpy as np

from bisect import bisect


def weighted_autocontrast(src_path, dst_path, white_perc, black_perc):
    img = cv2.imread(src_path, 0)
    assert img is not None

    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)

    w = np.cumsum(hist) - 0.5 * hist

    black_perc_val = min(255, bisect(np.cumsum(hist), float(black_perc)))
    white_perc_val = max(0, 255 - bisect(np.cumsum(reversed(hist)), float(white_perc)))

    def elem_weighted_autocontrast(val):
        if black_perc_val < white_perc_val:
            tmp_contrast_val = (w[val] - w[black_perc_val]) * (255.0 - 0.0) / (w[white_perc_val] - w[black_perc_val])
            return np.rint(max(0.0, min(255.0, tmp_contrast_val)))
        else:
            return val

    table = np.array([elem_weighted_autocontrast(val) for val in range(256)], dtype=np.uint8)

    assert cv2.imwrite(dst_path, cv2.LUT(img, table))


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    assert 0 <= float(argv[3]) < 1
    assert 0 <= float(argv[4]) < 1

    weighted_autocontrast(*argv[1:])

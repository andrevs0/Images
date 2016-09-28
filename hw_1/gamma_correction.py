from __future__ import print_function
from sys import argv
import os.path

import cv2
import math
import numpy as np


def gamma_correction(src_path, dst_path, a, b):
    img = cv2.imread(src_path)
    assert img is not None

    def elem_gamma_correction(val):
        val01 = val / 255.0
        gamma_val01 = max(0.0, min(1.0, float(a) * math.pow(val01, float(b))))
        gamma_val = np.rint(gamma_val01 * 255.0)
        return gamma_val

    table = np.array([elem_gamma_correction(val) for val in range(256)], dtype=np.uint8)

    assert cv2.imwrite(dst_path, cv2.LUT(img, table))


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])

    gamma_correction(*argv[1:])

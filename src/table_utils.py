from collections import OrderedDict
import numpy as np

from src.utils import line_equation_coefficients, line_intersection


def continue_lines(lines, img_height, img_width):
    horizontal = dict()
    vertical = dict()
    for i in range(0, len(lines)):
        line = lines[i][0]
        angel = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180. / np.pi

        if -91 < angel < -89:
            vertical[line[0]] = [line[0], 0, line[2], img_height]
        elif angel == 0:
            horizontal[line[1]] = [0, line[1], img_width, line[3]]

    horizontal = OrderedDict(sorted(horizontal.items()))
    vertical = OrderedDict(sorted(vertical.items()))

    h = reduce_lines(horizontal, img_height)
    print('ver----')
    v = reduce_lines(vertical, img_width)
    return h,v


def reduce_lines(lines, border, threshold=10):
    buckets = []
    prev = next(iter(lines.keys()))
    current_bucket = []

    for key in lines.keys():
        # Check if line is close to image border
        # than it would be the image border line which is not the table part
        # so skip them
        if key - 20 <= 0 or key + 20 > border:
            prev = key
            print(lines[key])
            continue

        if key > threshold + prev:

            buckets.append(current_bucket)
            current_bucket = []

        current_bucket.append(key)
        prev = key

    if len(current_bucket) > 0:
        buckets.append(current_bucket)

    result = []

    for bucket in buckets:
        if len(bucket) == 0:
            continue
        mean = np.mean(bucket)
        result.append(lines[min(bucket, key=lambda x: abs(x - mean))])

    return result


def find_line_intersection_points(horizontal, vertical):
    dots = []

    # Find intersection of horizontal and vertical lines and plot them on image (just for observation)
    for hor_line in horizontal:
        L1 = line_equation_coefficients(hor_line[:2], hor_line[2:])
        row = []
        for ver_line in vertical:

            L2 = line_equation_coefficients(ver_line[:2], ver_line[2:])
            inter = line_intersection(L1, L2)
            if inter:
                row.append(inter)

        dots.append(row)

    return dots

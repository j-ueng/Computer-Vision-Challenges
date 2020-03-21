import cv2 as cv
import numpy as np


def segment(image: np.ndarray) -> np.ndarray:

    z = image.reshape((-1, 3))
    z = np.float32(z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    ret, label, center = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = label.reshape((256, 256))

    edges = cv.Canny(image, 0, 60)
    for i in range(0, 255, 3):
        for j in range(0, 255, 3):
            blk = 0
            for x in range(0, 3):
                for y in range(0, 3):
                    if edges[i + x, j + y] == 0:
                        blk = blk + 1

            if blk < 9:
                for x in range(0, 3):
                    for y in range(0, 3):
                        edges[i + x, j + y] = 255

    for i in range(255, 0, -3):
        for j in range(255, 0, -3):
            blk = 0
            for x in range(0, 3):
                for y in range(0, 3):
                    if edges[i - x, j - y] == 0:
                        blk = blk + 1

            if blk < 9:
                for x in range(0, 3):
                    for y in range(0, 3):
                        edges[i - x, j - y] = 255

    color = 0
    for i in range(0, 256):
        for j in range(0, 256):
            if edges[i, j] == 0:
                color = labels[i, j]
                break
        break

    seg = np.zeros((256, 256))
    for i in range(0, 256):
        for j in range(0, 256):
            if edges[i, j] == 0:
                seg[i, j] = 1
            elif edges[i, j] == 255:
                if labels[i, j] == color:
                    seg[i, j] = 2
                elif labels[i, j] != color:
                    seg[i, j] = 3

    seg = np.float32(seg)
    image = cv.cvtColor(seg, cv.COLOR_GRAY2RGB)

    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

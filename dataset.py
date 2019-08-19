import cv2
import mahotas
import numpy as np

import imutils


def load_digits(dataset_path):
    data = np.genfromtxt(dataset_path, delimiter=',', dtype='uint8')
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)

    return data, target


def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)

    skew = moments['mu11'] / moments['mu02']

    M = np.float32([[1, skew, -0.5 * w * skew],
                    [0, 1, 0]])

    image = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    image = imutils.resize(image, width=width)

    return image


def center_extent(image, size):
    (e_w, e_h) = size

    if image.shape[1] > image.shape[0]:
        image = imutils.resize(image, width=e_w)
    else:
        image = imutils.resize(image, height=e_h)

    extent = np.zeros((e_h, e_w), dtype='uint8')

    offset_x = (e_w - image.shape[1]) // 2
    offset_y = (e_h - image.shape[0]) // 2
    extent[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1]] = image

    cm = mahotas.center_of_mass(extent)
    (c_y, c_x) = np.round(cm).astype('int32')
    (d_x, d_y) = ((size[0] // 2) - c_x, (size[1] // 2) - c_y)

    M = np.float32([[1, 0, d_x], [0, 1, d_y]])
    extent = cv2.warpAffine(extent, M, size)

    return extent

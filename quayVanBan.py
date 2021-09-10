import cv2
import numpy as np
from numpy.lib.function_base import angle
from scipy.ndimage import interpolation as inter

def get_angle(image, delta = 1, center = 0, range = 45):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape = False, order = 0)
        histogram = np.sum(data, axis = 1, dtype = object)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype = object)
        
        return histogram, score

    scores = []
    angles = np.arange(center - range, center + range + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    return best_angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def resize_image(image, max_width = 800, max_height = 1000):
    height, width = image.shape[:2]

    scale_with = float(max_width) / width
    scale_height = float(max_height) / height
    
    scale = max(scale_with, scale_height)

    if scale < 1.0:
        return cv2.resize(image, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
    else:
        return image

def correct_skew(image, delta = 0.01, range = 45):
    small = resize_image(image)
    
    _delta = 1
    angle = 0
    while _delta >= delta:
        angle = get_angle(small, _delta, angle, range)
        print(angle)

        range = _delta
        _delta /= 10

    return rotate_image(image, angle)


if __name__ == '__main__':
    image = cv2.imread('aa.jpg', 0)

    cv2.imwrite("res11.png", correct_skew(image))

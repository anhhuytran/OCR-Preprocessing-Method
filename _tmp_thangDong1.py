import cv2
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
import numpy as np

def findPointList(contours):
    res = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        res.append((cX, cY))

    return res

def get_next_contour(mask, contour):
    maxY, maxX = mask.shape

    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    current_contour = mask[cY][cX]
    while cX < maxX:
        if mask[cY][cX] != current_contour and mask[cY][cX] != 0:
            return current_contour, mask[cY][cX]
        cX += 1

    return current_contour, None

def connect_pair(pairs):
    for pair in pairs:
        return

img = cv2.imread('demo7.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 20)

revert = 255 - bin
dilate = cv2.dilate(revert, None, iterations=2)

contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours.sort(key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1])

check = [0] * len(contours)
lines = []
mask = np.zeros(gray.shape)
for i, contour in enumerate(contours):
    cv2.fillPoly(mask, pts =[contour], color= i + 1)

pair = []
for contour in contours:
    pair.append(get_next_contour(mask, contour))

print(pair)
lines = connect_pair(pair)

cv2.imwrite('abczyx.png', mask)
exit()







new = np.zeros(img.shape) + 255

cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
points = findPointList(contours)
points.sort()   


# #convert point list
# _points = zip(*points)
# _points = list(_points)

# f = UnivariateSpline(_points[0], _points[1])
# # f = CubicSpline(_points[0], _points[1])
# x_new = np.linspace(0, img.shape[1], img.shape[1] + 1)
# y_new = f(x_new)

# for point in points:
#     cv2.circle(img, point, radius=3, color=(0, 0, 255), thickness = -1)

# mask = np.zeros(gray.shape)
# pts = np.array(points, dtype=np.int32)
# cv2.polylines(img, [pts], False, 255, thickness=1, lineType=cv2.LINE_AA)

# mainAxis = mask.shape[0] / 2
# for x in range(0, mask.shape[1]):
#     delta = mainAxis - y_new[x]

#     for y in range(0, gray.shape[0]):
#         if bin[y][x] == 0:
#             new[y][x] = (200, 200, 200)
#             new[int(y + delta)][x] = 0

# # for y in range(0, mask.shape[1]):
# #     maskPoint = mainAxis
#     for x in range(0, mask.shape[0]):
#         if mask[x][y] == 255:
#             maskPoint = x
#             break

#     delta = mainAxis - maskPoint

#     for x in range(0, gray.shape[0]):
#         if bin[x][y] == 0:
#             new[int(x + delta)][y] = 0

cv2.imwrite('abczyx.png', mask)
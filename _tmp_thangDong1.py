import cv2
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
import numpy as np
import random

# FILE_NAME = 'demo6.jpg'

# DILATE_KERNEL = None        #np.ones((3,5), np.uint8)
# DILATE_LITERATION = 3
# MINIMUM_CONTOUR_SIZE = 100

# FIND_NEXT_WIDTH = 10
# FIND_NEXT_MAX_DISTANCE = 60
# FIND_NEXT_MAX_DEGREE = 45


FILE_NAME = 'demo2.jpg'

DILATE_KERNEL = np.ones((3,5), np.uint8)
DILATE_LITERATION = 4
MINIMUM_CONTOUR_SIZE = 500
MAXIMUM_CONTOUR_SIZE = 100000

FIND_NEXT_WIDTH = 10
FIND_NEXT_MAX_DISTANCE = 60
FIND_NEXT_MAX_DEGREE = 45


def center(M):
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def get_next_contour(mask, current_contour, M, delta, max_distance):
    h, w = mask.shape

    c_x = M["m10"] / M["m00"]
    c_y = M["m01"] / M["m00"]
    theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    alpha = np.tan(theta)
    angle = np.degrees(np.arctan(alpha))
    if angle < -FIND_NEXT_MAX_DEGREE or angle > FIND_NEXT_MAX_DEGREE:
        return None, -1
    
    x = c_x
    y = c_y
    delta = delta / np.cos(np.arctan(alpha))
    while pow(pow(x - c_x, 2) + pow(y - c_y, 2), 0.5) < max_distance:
        x += 1
        y += alpha

        _x = int(x)
        if _x < 0 or _x >= w:
            break

        for _y in range(int(y - delta), int(y + delta + 1)):
            if _y < 0 or _y >= h:
                break

            ### draw line
            img_fncnt[_y][_x] += 200

            if mask[_y][_x] != 0:
                if int(mask[_y][_x]) - 1 != current_contour:
                    return int(mask[_y][_x]) - 1, pow(pow(x - c_x, 2) + pow(y - c_y, 2), 0.5)

                c_x = x
                c_y = y

    return None, -1

def connect_pair(pairs):
    chains = []
    
    check = [False] * len(pairs)
    i = 0
    while True:
        while i < len(pairs) and check[i] is not False:
            i += 1
        if i == len(pairs):
            break

        chain = [i]
        check[i] = len(chains)
        while True:
            print(chain)

            next = pairs[chain[-1]]
            if next is None:
                chain.append(None)
                break
            else:
                if check[next] is False:
                    check[next] = True
                    chain.append(next)
                else:
                    try:
                        # check[next] is the index of chain have begin element is next
                        chains[check[next]] = chain + chains[check[next]]
                        check[chain[0]] = check[next]
                        check[next] = True

                        chain = []
                        break 
                    except IndexError:
                        print(check)
                        print(check[next], next)
                        raise IndexError

        if len(chain) > 1:
            chains.append(chain)
    return chains

def check_size(contour):
    area = cv2.contourArea(contour)
    return area > MINIMUM_CONTOUR_SIZE and area < MAXIMUM_CONTOUR_SIZE

global img_fncnt

# read image and threshold
img = cv2.imread(FILE_NAME)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 20)

img_cnt = img.copy()
img_fncnt = img.copy()

# reduce noise ?
# connect small component
revert = 255 - bin
dilate = cv2.dilate(revert, kernel=DILATE_KERNEL, iterations=DILATE_LITERATION)

#find contours and remove small contours
contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = list(filter(lambda contour: check_size(contour), contours))

### drawing contour with index
for contour_index, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contours[contour_index])
    cv2.rectangle(img_cnt, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(img_cnt, str(contour_index), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)


cv2.imwrite('result_dilate.png', dilate)
cv2.imwrite('result_cnt.png', img_cnt)
exit()



# create mask of contours
mask = np.zeros(bin.shape)
for i, contour in enumerate(contours):
    cv2.fillPoly(mask, pts = [contour], color = i + 1)

# calculate moment of contours
contours_moments = []
for contour in contours:
    contours_moments.append(cv2.moments(contour))

# get pair of 2 words
pairs = [None] * len(contours)
contours_distance = [[-1, 0]] * len(contours)
for i, contour in enumerate(contours):
    next_contour_index, distance = get_next_contour(mask, i, contours_moments[i], FIND_NEXT_WIDTH, FIND_NEXT_MAX_DISTANCE)
    if next_contour_index is not None:
        if contours_distance[next_contour_index][0] == -1:
            pairs[i] = next_contour_index
            contours_distance[next_contour_index] = i, distance
        
        elif distance < contours_distance[next_contour_index][1]:
            pairs[contours_distance[next_contour_index][0]] = None

            pairs[i] = next_contour_index
            contours_distance[next_contour_index] = i, distance


for i, pair in enumerate(pairs):
    print(i, pair)

# connect pair to chains
chains = connect_pair(pairs)
print(chains)

lines = []
for chain in chains:
    chain.pop()
    if len(chain) < 2:
        lines.append((-1, -1))
        continue
    
    [x, y, w, h] = cv2.boundingRect(contours[chain[0]])
    mainAxis = y + int(h / 2)

    # caculate center point list for each chain
    center_points= []
    for contour_index in chain:
        center_points.append(center(contours_moments[contour_index]))

    ### drawing center point on original image
    color1 = list(np.random.choice(range(256), size=3))
    color =[int(color1[0]), int(color1[1]), int(color1[2])] 
    for i, contour_index in enumerate(chain):
        cv2.circle(img_fncnt, center_points[i], radius=10, color = color, thickness=-1)
        cv2.putText(img_fncnt , str(contour_index), center_points[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 100), 2)

    
    # calculating CubicSpline for each chain
    center_points.sort()
    _center_points = zip(*center_points)
    _center_points = list(_center_points)
    lines.append((mainAxis, CubicSpline(_center_points[0], _center_points[1])))
    
# create list store place of contour in chains
pos = [-1] * len(contours)
for i, chain in enumerate(chains):
    for contour_index in chain:
        pos[contour_index] = i

# create result image
print('Starting create new image...')
new = np.zeros(bin.shape) + 255
h, w = new.shape

for y in range(0, h):
    for x in range(0, w):
        if bin[y][x] == 0 and mask[y][x] != 0: 
            mainAxis, f = lines[pos[int(mask[y][x] - 1)]]
            if mainAxis == -1:
                continue

            new_y = int(y + mainAxis - f(x))
            if 0<= new_y < h:
                new[new_y][x] = 0

print('Done')

cv2.imwrite('result_fnct.png', img_fncnt)
cv2.imwrite('result.png', new)
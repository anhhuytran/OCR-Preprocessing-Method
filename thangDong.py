import enum
import cv2
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
import numpy as np
import random
from operator import add, sub

FILE_NAME = 'demo/demo1.png'

DILATE_KERNEL = None  #np.ones((3,5), np.uint8)
DILATE_LITERATION = 3
MIN_CONTOUR_SIZE = 80
MAX_CONTOUR_SIZE = 100000
MAX_CONTOUR_WIDTH = 1000
MAX_CONTOUR_HEIGHT = 500

FIND_NEXT_WIDTH = 10
FIND_NEXT_MAX_DISTANCE = 60
FIND_NEXT_MAX_DEGREE = 30

MINIMUM_LEN_SPLINE = 4
SPLINE_S = 1000


def center(M):
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def linear_function(M):
    c_x, c_y = center(M)
    theta = 0.5 * np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
    alpha = np.tan(theta)
    angle = np.degrees(np.arctan(alpha))
    if angle < -FIND_NEXT_MAX_DEGREE or angle > FIND_NEXT_MAX_DEGREE:
        return None

    return lambda x : c_y + alpha * (x - c_x)
    
def get_next_contour(mask, current_contour, M, delta, max_distance):
    h, w = mask.shape

    c_x, c_y = center(M)
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
        while True:
            next = pairs[chain[-1]]
            if next is None:
                break

            else:
                if check[next] is False:
                    check[next] = True
                    chain.append(next)
                else:
                    # check[next] is the index of chain have begin element is next
                    chains[check[next]] = chain + chains[check[next]]
                    check[chain[0]] = check[next]
                    check[next] = True

                    chain = []
                    break 

        if len(chain) > 0:
            check[i] = len(chains)
            chains.append(chain)
    return chains

def check_size(contour):
    area = cv2.contourArea(contour)
    [x, y, w, h] = cv2.boundingRect(contour)

    return area > MIN_CONTOUR_SIZE and area < MAX_CONTOUR_SIZE and w < MAX_CONTOUR_WIDTH and h < MAX_CONTOUR_HEIGHT

def find_function(chain, contours_moments):
    center_points= []
    for contour_index in chain:
        center_points.append(center(contours_moments[contour_index])) 
    
    # calculating Spline for chain
    center_points.sort()
    i = 0
    while i < len(center_points):
        if center_points[i][0] == center_points[i-1][0]:
            center_points.pop(i)
        else:
            i += 1
    if len(center_points) < 2:
        return None

    _center_points = zip(*center_points)
    _center_points = list(_center_points)
    if len(center_points) < MINIMUM_LEN_SPLINE:
        return CubicSpline(_center_points[0], _center_points[1])
    return UnivariateSpline(_center_points[0], _center_points[1], s=SPLINE_S)

def len_route(route):
    sum = 0.0
    for i in range(1, len(route)):
        sum += pow((route[i][0] - route[i-1][0])**2 + (route[i][1] - route[i-1][1])**2, 0.5)

    return sum

def find_neighbor(mask, current_contour, begin_point, f, x_function = add):
    h, w = mask.shape

    x = begin_point[0]
    route = []
    while len_route(route) < FIND_NEXT_MAX_DISTANCE:
        x = x_function(x, 1)
        y = f(x)
        route.append((x, float(y)))

        if len(route) < 2:
            delta = FIND_NEXT_WIDTH
        else:
            dx = route[-1][0] - route[-2][0]
            dy = route[-1][1] - route[-2][1]
            delta = FIND_NEXT_WIDTH / np.cos(np.arctan(dy / dx))


        _x = int(x)
        if _x < 0 or _x >= w:
            break

        for _y in range(int(y - delta), int(y + delta + 1)):
            if _y < 0 or _y >= h:
                break
            
            ### draw line
            # img_fncnt[_y][_x] += 100

            if mask[_y][_x] != 0:
                if int(mask[_y][_x]) - 1 != current_contour:
                    return int(mask[_y][_x]) - 1, len_route(route)

                route.clear()

    return None, -1

def drawLine(img, f, color, begin = None, end = None):
    h, w, _ = img.shape
    if end is None:
        begin = 0
        end = w

    for x in range(begin, end):
        y = int(f(x))
        if 0 <= y < h:
            img[y][x] = color

def find_closest_element(mask, chain, contours_moments):
    f = find_function(chain, contours_moments)
    if f is None:
        return None, None
    
    prev_element = find_neighbor(mask, chain[0], center(contours_moments[chain[0]]), f, sub)
    next_element = find_neighbor(mask, chain[-1], center(contours_moments[chain[-1]]), f, add)
    return prev_element, next_element
    


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
# exit()



# create mask of contours
mask = np.zeros(bin.shape)
for i, contour in enumerate(contours):
    cv2.fillPoly(mask, pts = [contour], color = i + 1)

# calculate moment of contours
contours_moments = []
for contour in contours:
    contours_moments.append(cv2.moments(contour))

# get pair of 2 words
print('Find pairs')
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

# ## print pairs
# for i, pair in enumerate(pairs):
#     print(i, pair)

# connect pair to chains
print('Connect pairs...')
chains = connect_pair(pairs)
# ## print chains
# for chain in chains:
#     print(chain)

# extend the long chains
print('Extend the chains with isolate...')
long_chains = []
isolate_contours = []
for chain in chains:
    if len(chain) > 1:
        long_chains.append(chain)
    else:
        isolate_contours.append(chain + [-1] * 3)

# isolate_contours -> 0: contour_index, 1: chain_index, 2: distance, 3: type
# add isolate element to chain
long_chains_done = []
closed_contours = [None] * len(long_chains)
while True:
    i = 0
    while i < len(long_chains):
        if closed_contours[i] is None:
            prev_cnt, next_cnt = find_closest_element(mask, long_chains[i], contours_moments)
            if prev_cnt[0] is None and next_cnt[0] is None:
                long_chains_done.append(long_chains[i])
                long_chains.pop(i)
                closed_contours.pop(i)
                continue
            else:
                closed_contours[i] = (prev_cnt, next_cnt)
        i += 1

    for chain_index, closed_contours_pair in enumerate(closed_contours):
        for i, closed_contour in enumerate(closed_contours_pair):
            if closed_contour[0] is None:
                continue

            for contour in isolate_contours:
                if closed_contour[0] == contour[0]:
                    if contour[2] == -1 or closed_contour[1] < contour[2]:
                        contour[1] = chain_index
                        contour[2] = closed_contour[1]
                        contour[3] = i
                    break
    
    i = 0
    have_insert = False
    while i < len(isolate_contours):
        if isolate_contours[i][1] != -1:
            if isolate_contours[i][3] == 0:
                long_chains[isolate_contours[i][1]].insert(0, isolate_contours[i][0])
            else:
                long_chains[isolate_contours[i][1]].append(isolate_contours[i][0])
            closed_contours[isolate_contours[i][1]] = None
            isolate_contours.pop(i)
            have_insert = True
        else:
            i += 1
    if not have_insert:
        break

# for i, chain in enumerate(long_chains):
#     print(i ,': ', chain, closed_contours[i])

# connect chains together
print('Connect chains together')   
connect_chain = []
for i in range(0, len(long_chains)):
    connect_chain.append([[None, -1], [None, -1]])        
for chain_index, closed_contour in enumerate(closed_contours):
    for t, contour in enumerate(closed_contour):
        if contour[0] is None:
            continue

        for i, chain in enumerate(long_chains):
            if contour[0] == chain[t - 1]:
                if connect_chain[i][t - 1][1] == -1 or contour[1] < connect_chain[i][t - 1][1]:
                    connect_chain[i][t - 1][0] = chain_index
                    connect_chain[i][t - 1][1] = contour[1]
                break

while True:
    i = 0
    while i < len(connect_chain) and connect_chain[i][1][0] is None:
        i+= 1
    if i == len(connect_chain):
        break

    next_chain_index = connect_chain[i][1][0]
    if connect_chain[next_chain_index][0][0] == i:
        long_chains[i].extend(long_chains[next_chain_index])

        connect_chain[i][1] = connect_chain[next_chain_index][1]
        next_next_chain_index = connect_chain[i][1][0]
        if next_next_chain_index is not None and connect_chain[next_next_chain_index][0][0] == next_chain_index:
            connect_chain[next_next_chain_index][0][0] = i

        connect_chain[next_chain_index][1] = [None, -2]  # -2 the hien da gan vao 1 chain khac
    else:
        connect_chain[i][1] = [None, -1]

i = 0
while i < len(long_chains):
    if connect_chain[i][1][1] == -2:
        long_chains.pop(i)
        connect_chain.pop(i)
    else:
        i += 1

chains = long_chains_done + long_chains

lines = []
for chain in chains:
    ### drawing center point on original image
    center_points= []
    for contour_index in chain:
        center_points.append(center(contours_moments[contour_index]))
    color1 = list(np.random.choice(range(256), size=3))
    color =[int(color1[0]), int(color1[1]), int(color1[2])] 
    for i, contour_index in enumerate(chain):
        cv2.circle(img_fncnt, center_points[i], radius=10, color = color, thickness=-1)
        cv2.putText(img_fncnt , str(contour_index), center_points[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 100), 2)
    
    # calculating CubicSpline for each chain
    f = find_function(chain, contours_moments)
    lines.append([-1, f])

    ### drawing a line
    drawLine(img_fncnt, f, color)


cv2.imwrite('result_fnct.png', img_fncnt)

# draw isolate contour
for contour_index, _, _, _ in isolate_contours:
    f = linear_function(contours_moments[contour_index])
    if f is not None:
        chains.append([contour_index])
        lines.append([-1, f])

# add the start point (main_axis)
for i, line in enumerate(lines):
    f = line[1]
    begin_contour_index = chains[i][0]
    
    x = center(contours_moments[begin_contour_index])[0]
    while True:
        x -= 1
        y = f(x)

        _x, _y = int(x), int(y)
        main_axis = _y
        if _x < 0 or _x >= w or _y < 0 or _y >= h or mask[_y][_x] == 0:
            break
    line[0] = main_axis
            

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
            chain_index = pos[int(mask[y][x] - 1)]
            if chain_index != -1:
                mainAxis, f = lines[chain_index]
                new_y = int(y + mainAxis - f(x))
                if 0<= new_y < h:
                    new[new_y][x] = 0
            else:
                pass

print('Done')

cv2.imwrite('result.png', new)
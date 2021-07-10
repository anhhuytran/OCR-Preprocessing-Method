# from PIL import Image
# import pytesseract
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# # print(pytesseract.image_to_string(Image.open('test_images/captured/binarized/' + '011' + '_SCANNED_BIN_GT.png')))
# # print(pytesseract.image_to_boxes(Image.open('test_images/captured/binarized/' + '011' + '_SCANNED_BIN_GT.png')))

# # pdf = pytesseract.image_to_pdf_or_hocr('test_images/captured/binarized/' + '011' + '_SCANNED_BIN_GT.png', extension='pdf')
# # with open('test.pdf', 'w+b') as f:
# #     f.write(pdf) # pdf type is bytes by default
# import cv2
# img = cv2.imread('test_images/captured/grayscale/' + '001' + '_CC_GRAY.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('a', img)
# cv2.waitKey(0)






import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=1, limit=45):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score /1000000000

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated, scores



if __name__ == '__main__':
    image = cv2.imread('demo3.png')
    angle, rotated, score = correct_skew(image, 5)
    print(score)
    print(angle)


    # grab the dimensions of the image and calculate the center of the
    # # image
    # (h, w) = image.shape[:2]
    # (cX, cY) = (w // 2, h // 2)
    # # rotate our image by 45 degrees around the center of the image
    # M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # rotated = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite("Rotated by 45 Degrees.png", rotated)









    
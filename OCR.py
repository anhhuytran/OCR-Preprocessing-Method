from PIL import Image
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# print(pytesseract.image_to_string(Image.open('test_images/captured/binarized/' + '011' + '_SCANNED_BIN_GT.png')))
# print(pytesseract.image_to_boxes(Image.open('test_images/captured/binarized/' + '011' + '_SCANNED_BIN_GT.png')))

# pdf = pytesseract.image_to_pdf_or_hocr('test_images/captured/binarized/' + '011' + '_SCANNED_BIN_GT.png', extension='pdf')
# with open('test.pdf', 'w+b') as f:
#     f.write(pdf) # pdf type is bytes by default
import cv2
img = cv2.imread('test_images/captured/grayscale/' + '001' + '_CC_GRAY.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('a', img)
cv2.waitKey(0)

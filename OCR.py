from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
print(pytesseract.image_to_string(Image.open('test_images/007.jpg')))
#print(pytesseract.image_to_boxes(Image.open('test7.png')))
#print(pytesseract.image_to_data(Image.open('test7.png')))

# pdf = pytesseract.image_to_pdf_or_hocr('test7.png', extension='pdf')
# with open('test.pdf', 'w+b') as f:
#     f.write(pdf) # pdf type is bytes by default
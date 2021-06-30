from PIL import Image

# function that return average value of pixels
def pixelsAVGVal(pixels):
    return int(sum(pixels) / len(pixels))

# function that return threshold value of a gray-scale image
# 1. Select initial threshold value, typically the mean 8-bit value of the original image.
# 2. Divide the original image into two portions;
    # 2.1 Pixel values that are less than or equal to the threshold; background
    # 2.2 Pixel values greater than the threshold; foreground
# 3. Find the average mean values of the two new images
# 4. Calculate the new threshold by averaging the two means.
# 5. If the difference between the previous threshold value and the new threshold value are below a specified limit, you are finished. Otherwise apply the new threshold to the original image keep trying.
def findThresholdVal(imgPath):
    img = Image.open(imgPath, 'r')
    pixels = list(img.getdata())

    thresholdVal = pixelsAVGVal(pixels)
    foreground = list()
    background = list()

    while True:
        for i in range(img.height):
            for j in range(img.width):
                if(pixels[i*j+j]<=thresholdVal):
                    background.append(pixels[i*j+j])
                else:
                    foreground.append(pixels[i*j+j])

        AVGVal = int((pixelsAVGVal(foreground) + pixelsAVGVal(background)) / 2)
        # print(AVGVal)
        foreground.clear()
        background.clear()

        if(abs(AVGVal-thresholdVal) > 1):
            # print(thresholdVal)
            thresholdVal = AVGVal
        else:
            break

    return thresholdVal

def binarizeImage(srcImgPath, dstImgPath):
    img = Image.open(srcImgPath, 'r')
    thresholdVal = findThresholdVal(srcImgPath)
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if(pixels[i, j] < thresholdVal):
                pixels[i, j] = 0
            else:
                pixels[i, j] = 255

    img.save(dstImgPath)


binarizeImage('test_images/007.jpg', 'test_images/007_binary.jpg')
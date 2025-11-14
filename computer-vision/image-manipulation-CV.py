from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2

FILE_PATH = '/Users/fpj/Development/python/ml-with-python/computer-vision/data/'
BABOON_FILE = FILE_PATH + 'baboon.png'
CAT_FILE = FILE_PATH + 'cat.png'
LENNA_FILE = FILE_PATH + 'lenna.png'
BARBARA_FILE = FILE_PATH + 'barbara.png'

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

my_image = FILE_PATH + 'lenna.png'
image = cv2.imread(my_image, cv2.IMREAD_UNCHANGED)
print("OpenCV image : ", type(image))
# show the max and min intensity values of the image
print("Max intensity: ", image.max())
print("Min intensity: ", image.min())
#
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image gray', image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# work with color channels
baboon=cv2.imread(BABOON_FILE)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()
#
blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]
#
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(122)
plt.imshow(im_bgr,cmap='gray')
plt.title("Different color channels  blue (top), green (middle), red (bottom)  ")
plt.show()

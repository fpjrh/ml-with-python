from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = '/Users/fpj/Development/python/ml-with-python/computer-vision/data/'
BABOON_FILE = FILE_PATH + 'baboon.png'
CAT_FILE = FILE_PATH + 'cat.png'
LENNA_FILE = FILE_PATH + 'lenna.png'
BARBARA_FILE = FILE_PATH + 'barbara.png'

baboon = np.array(Image.open(BABOON_FILE))
plt.figure(figsize=(5,5))
plt.imshow(baboon)
plt.show()

# let's play with the cat image
image = Image.open(CAT_FILE)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()
#
array = np.array(image)
width, height, C = array.shape
print('width, height, C', width, height, C)
# flip the image
array_flip = np.zeros((width, height, C), dtype=np.uint8)
#
for i, row in enumerate(array):
    array_flip[width - i - 1, :, :] = row
# let's use ImageOps to flip the image
image_flip = ImageOps.flip(image)
plt.figure(figsize=(10, 10))
plt.imshow(image_flip)
plt.show()
#
image_mirror = ImageOps.mirror(image)
plt.figure(figsize=(10, 10))
plt.imshow(image_mirror)
plt.show()



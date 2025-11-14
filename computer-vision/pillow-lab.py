from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = '/Users/fpj/Development/python/ml-with-python/computer-vision/data/'

# define a helper function to concatenate two images side-by-side

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

my_image = FILE_PATH + 'lenna.png'

image = Image.open(my_image)
print("PIL image : ", type(image))

# show the image
#image.show()

# show the image with matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

# print the image dimensions
print("Image dimensions: ", image.size)
print("Image height: ", image.height)
print("Image width: ", image.width)
# print the image mode
print("Image mode: ", image.mode)
# print the image format
print("Image format: ", image.format)

# load the image into memory
im = image.load()

# check the intensity of a pixel
print("Intensity of a pixel at (100, 100): ", im[100, 100])

# convert the image to grayscale using ImageOps
image_gray = ImageOps.grayscale(image)
# show the grayscale image
plt.figure(figsize=(10, 10))
plt.imshow(image_gray, cmap='gray')
plt.show()

# let's quantize the image
for n in range(3,8):
    plt.figure(figsize=(10,10))

    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n))) 
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
    plt.show()
#

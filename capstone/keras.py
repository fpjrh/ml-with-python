import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork

from PIL import Image

negative_dir = '/Users/fpj/Development/python/ml-with-python/capstone/resources/data/Negative'
negative_files = os.scandir(negative_dir)
print ("Negative files iterator : ", negative_files)
#
neg_file_01 = next(negative_files)
print("First negative file: ", neg_file_01)
#
image_name = str(neg_file_01).split("'")[1]
print("Image name: ", image_name)
# read in the image data
fmt_dir_string = negative_dir + '/{}'
image_data = plt.imread(fmt_dir_string.format(image_name))
print("Image data: ", image_data)

# Get the dimensions of the image
image_height, image_width, image_channels = image_data.shape
# Print the image size
print("Image Size: {}".format(image_data.shape))

# list of negative images
negative_images = os.listdir(negative_dir)
print("Number of negative images: ", len(negative_images))

# sort the negative images
negative_images.sort()
print("First three negative images: ", negative_images[0:3])

# Display images 2 through 5
for i in range(1, 5):
    image_name = negative_images[i]
    image_data = plt.imread(fmt_dir_string.format(image_name))
    plt.imshow(image_data)
    plt.title('Image: {}'.format(image_name))
    plt.show()
#
# get the positive images
positive_dir = '/Users/fpj/Development/python/ml-with-python/capstone/resources/data/Positive'
positive_images = os.listdir(positive_dir)
positive_images.sort()  
pos_fmt_dir_string = positive_dir + '/{}'
print("Number of positive images: ", len(positive_images))
# show the first four images
for i in range(0, 4):
    image_name = positive_images[i]
    image_data = plt.imread(pos_fmt_dir_string.format(image_name))
    plt.imshow(image_data)
    plt.title('Image: {}'.format(image_name))
    plt.show()

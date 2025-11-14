import matplotlib.pyplot as plt
import numpy as np
import cv2

# Image constants
IMG_PATH = '/Users/fpj/Development/python/ml-with-python/computer-vision/data/'
IMG_LENNA = 'lenna.png'
IMG_BARBARA = 'barbara.png'
IMG_CAMERAMAN = 'cameraman.jpeg'

# define a function to plot two images side by side
def plot_image(image1, image2, title_1="Original", title_2="Edited"):
    """
    Plots two images side by side with their respective titles.
    
    Args:
        image1 (numpy.ndarray): The first image to be plotted.
        image2 (numpy.ndarray): The second image to be plotted.
        title1 (str): The title for the first image.
        title2 (str): The title for the second image.
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()
    
# lod the lenna image and display it
#image = cv2.imread(IMG_PATH + IMG_LENNA)
#print(image)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.show()
#
# Get the number of rows and columns in the image
#rows, cols,_= image.shape
# Creates values using a normal distribution with a mean of 0 
# and standard deviation of 15, the values are converted to unit8 which means the values are between 0 and 255
#noise = np.random.normal(0,15,(rows,cols,3)).astype(np.uint8)
# Add the noise to the image
#noisy_image = image + noise
# Plots the original image and the image with noise using the function defined at the top
#plot_image(image, noisy_image, title_1="Orignal",title_2="Image Plus Noise")
# create a kernel which is a 6 by 6 array where each value is 1/36
#kernel = np.ones((6, 6), np.float32) / 36
# Filters the images using the kernel
#image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)
# Plots the Filtered and Image with Noise using the function defined at the top
#plot_image(image_filtered, noisy_image,title_1="Filtered image",title_2="Image Plus Noise")
# Creates a kernel which is a 4 by 4 array where each value is 1/16
kernel = np.ones((4,4))/16
# Filters the images using the kernel
#image_filtered=cv2.filter2D(src=noisy_image,ddepth=-1,kernel=kernel)
# Plots the Filtered and Image with Noise using the function defined at the top
#plot_image(image_filtered , noisy_image,title_1="filtered image",title_2="Image Plus Noise")

# Filters the images using GaussianBlur on the image with noise using a 4 by 4 kernel 
#image_filtered = cv2.GaussianBlur(noisy_image,(5,5),sigmaX=4,sigmaY=4)
# Plots the Filtered Image then the Unfiltered Image with Noise
#plot_image(image_filtered , noisy_image,title_1="Filtered image",title_2="Image Plus Noise")

# Filters the images using GaussianBlur on the image with noise using a 11 by 11 kernel 
#image_filtered = cv2.GaussianBlur(noisy_image,(11,11),sigmaX=10,sigmaY=10)
# Plots the Filtered Image then the Unfiltered Image with Noise
#plot_image(image_filtered , noisy_image,title_1="filtered image",title_2="Image Plus Noise")

# Common Kernel for image sharpening
kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
# Applys the sharpening filter using kernel on the original image without noise
#sharpened = cv2.filter2D(image, -1, kernel)
# Plots the sharpened image and the original image without noise
#plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

# Load the barbara image and display it
img_gray = cv2.imread(IMG_PATH + IMG_BARBARA)
print(img_gray)
plt.imshow(img_gray, cmap='gray')
plt.show()

# Filters the images using GaussianBlur on the image with noise using a 3 by 3 kernel 
img_gray = cv2.GaussianBlur(img_gray,(3,3),sigmaX=0.1,sigmaY=0.1)
# Renders the filtered image
plt.imshow(img_gray ,cmap='gray')
plt.show()
#
kernel = np.array([[1, 0, -1], 
                   [2, 0, -2],
                   [1, 0, -1]])
# approximate the derivative of the image using the sobel filter
ddepth = cv2.CV_16S
# apply the filter in the x direction
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
# plot the result
plt.imshow(grad_x, cmap='gray')
plt.show()
#
# apply the filter in the y direction
grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
# plot the result
plt.imshow(grad_y, cmap='gray')
plt.show()
# convert the values back to a number between 0 and 255
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
# add the derivative in the x and y direction together
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# make the figure bigger and plot the results
plt.figure(figsize=(10, 10))
plt.imshow(grad, cmap='gray')
plt.show()

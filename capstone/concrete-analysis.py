from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset
import skillsnetwork

# function to plot
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
#

directory="/Users/fpj/Development/python/ml-with-python/capstone/resources/data"
negative='Negative'
negative_file_path=os.path.join(directory,negative)
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
print("First three negative files: ", negative_files[0:3])
#
positive='Positive'
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
print("First three positive files: ", positive_files[0:3])
#

# Find the combined length of the list positive_files and negative_files
number_of_samples = len(positive_files) + len(negative_files)
print("Number of samples: ", number_of_samples)
#
# Assign labels to images
#
Y=torch.zeros([number_of_samples])
#
Y=Y.type(torch.LongTensor)
print("Y: ", Y)
# Set the even elements in the tensor to class 1 and the odd elements to class 0
Y[::2]=1
Y[1::2]=0
#
# Create a list all_files such that the even indexes contain the path to a positive image and the odd indexes contain the path to a negative image
#
all_files = []
#
for idx in range(0, number_of_samples, 2):
    all_files.append(positive_files[idx // 2])
    all_files.append(negative_files[idx // 2])
#
print("First six files: ", all_files[5000:6])
print("Length of all_files: ", len(all_files))
# show some samples
for y,file in zip(Y, all_files[500:4]):
    plt.imshow(Image.open(file))
    plt.title("y="+str(y.item()))
    plt.show()
#
#for y,file in zip(Y, all_files[30000:4]):
#    plt.imshow(Image.open(file))
#    plt.title("y="+str(y.item()))
#    plt.show()
train_files = all_files[0:30000]
#
val_files = all_files[30000:]
#
print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
#
# Create a dataset object for training and validation
#
#

# Create a dataset object for training and validation
#
#dataset = Dataset(train=True)
#print("Length of dataset: ", len(dataset))
#
# Get the 10th and 100th sample
#
#print("10th sample: ", dataset[9], type(dataset[9]))
#print("100th sample: ", dataset[99], type(dataset[99]))
#

#show_data(dataset[9][0])
#show_data(dataset[99][0])

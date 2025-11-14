# These are the libraries will be used for this lab.
import torchvision.models as models
from torchvision import transforms
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
#
import h5py
import os
import glob
import time
#
torch.manual_seed(0)
#
# See what device we are running on
if torch.cuda.is_available():
    print("Using the Cuda")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using the MPS")
    # MPS is only available on macOS
else:
    print("Using the CPU")
    device = torch.device("cpu")
#
#
print("Using device: ", device)
#
# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="/Users/fpj/Development/python/ml-with-python/capstone/data/resnet-18"
        positive="Positive_tensors"
        negative='Negative_tensors'

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)     
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
               
        image=torch.load(self.all_files[idx], weights_only=True)
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

# Create two dataset objects, one for traing and one for validation
train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
print("Dataset objects created")
#
# QUESTION 1
#
# Load the pre-trained model resnet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

print("Model loaded")
# set the parameter requires_grad to false for all the parameters
for param in model.parameters():
    param.requires_grad = False
# Replace the output layer model.fc of the neural network with a nn.Linear object, to classify 2 different classes.
model.fc = nn.Linear(512, 2)
print("Output layer replaced")
# set model to device
model.to(device)
# print out the model structure
print(model)
#
# QUESTION 2
#
# Create a loss function
criterion = nn.CrossEntropyLoss()
# create a training loader and validation loader object with a batch size of 100 samples each
train_loader = DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=100)
# create an optimizer object
optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)
#
#
n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:
        # set x and y to device
        x, y = x.to(device), y.to(device)
        model.train() 
        #clear gradient
        optimizer.zero_grad()
        #make a prediction 
        z = model(x)
        # calculate loss
        loss = criterion(z, y) 
        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step() 
        #
        loss_list.append(loss.item())
    correct=0
    for x_test, y_test in validation_loader:
        # set x and y to device
        x_test, y_test = x_test.to(device), y_test.to(device)
        # set model to eval 
        model.eval()
        #make a prediction 
        z = model(x_test)
        #find max 
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
   
    accuracy=correct/N_test

#
# Print out the time it took to train the model
#
print("Training time: ", time.time()-start_time, " seconds")
# Print out the accuracy of the model
print("Accuracy: ", accuracy)
# Print out the loss of the model
print("Loss: ", loss.item())
# Print out the loss and accuracy list
print("Loss list: ", loss_list)
print("Accuracy list: ", accuracy_list)

plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# find the first four mis-classified samples

def show_data(data_sample):
    plt.imshow(data_sample[0])
    plt.title('y = ' + str(data_sample[1]))

count = 0
idx_count = 0
for x, y in validation_dataset:
    x = torch.unsqueeze(x, 0)
    x, y = x.to(device), y.to(device)
    model.eval()
    z = model(x)
    _, yhat = torch.max(z.data, 1)
    #
    if yhat != y:
        print("idx_count: ", idx_count, " - y: ", y.item(), " - yhat: ", yhat.item())
        #print('Real: ',y.item(),' - Predicted: ',yhat.item() )
        #plt.imshow(transforms.ToPILImage()(x[0]), interpolation="bicubic")
        #plt.show()
        count += 1
    idx_count += 1
    if count >= 5:
        break
#
print("We're all done here")

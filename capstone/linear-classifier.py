from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim 
import skillsnetwork 

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="/Users/fpj/Development/python/ml-with-python/capstone/resources/data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
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
            self.all_files=self.all_files[0:10000] #Change to 30000 to use the full test dataset
            self.Y=self.Y[0:10000] #Change to 30000 to use the full test dataset
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
        
        
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]
          
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
#
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# Create a transform object that uses the Compose function. First use the transform TOTensor90 and followed by Normalize(mean, std)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
# Create a dataset object for training and validation
dataset_train = Dataset(transform=transform, train=True)
dataset_val = Dataset(transform=transform, train=False)
# print the shape of the training dataset
print("Shape of the training dataset: ", dataset_train[0][0].shape)
# 
size_of_image=3*227*227
print("Size of the image: ", size_of_image)
# 
# Training section
#
learning_rate = 0.1
momentum = 0.1
batch_size = 5
epochs = 5
torch.manual_seed(0)
loss_function = nn.CrossEntropyLoss()
#
# Create a custom module called model for Softmax for two classes; input size should be size_of_image
#
class Softmax(torch.nn.Module):
    "custom softmax module"
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
 
    def forward(self, x):
        pred = self.linear(x)
        return pred
#
# Create a model object for Softmax for two classes; input size should be size_of_image
#
model = Softmax(size_of_image, 2)
#
# Create a optimizer object for SGD
#
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# Criterion for loss
#
# Create a criterion object for CrossEntropyLoss
#
criterion = nn.CrossEntropyLoss()

# Create a dataloader object for training and validation
#
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size)
# Train the model for 5 epochs
#
for epoch in range(epochs):
    # For each batch in the training loader
    for x, y in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        # Get the predictions
        z = model(x.view(-1, size_of_image))
        # Calculate the loss
        loss = criterion(z, y)
        # Calculate the accuracy
        _, yhat = torch.max(z, 1)
        # Calculate the gradients
        loss.backward()
        # Update the weights
        optimizer.step()
    # Print the loss
    print("Epoch: ", epoch, "Loss: ", loss.item())
   
#

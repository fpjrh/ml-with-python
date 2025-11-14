import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn,optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# set the random seed
torch.manual_seed(1)
#
from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self):
            self.x=torch.zeros(20,2)
            self.x[:,0]=torch.arange(-1,1,0.1)
            self.x[:,1]=torch.arange(-1,1,0.1)
            self.w=torch.tensor([ [1.0,-1.0],[1.0,3.0]])
            self.b=torch.tensor([[1.0,-1.0]])
            self.f=torch.mm(self.x,self.w)+self.b
            
            self.y=self.f+0.001*torch.randn((self.x.shape[0],1))
            self.len=self.x.shape[0]

    def __getitem__(self,index):

        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
# create a data object
data_set = Data()

# create a custom module
class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
# create a model
model = linear_regression(2, 2)
# create an optimizer with a learning rate of 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)
# create a loss function
criterion = nn.MSELoss()
# create a data loader object with a batch size of 5
train_loader = DataLoader(dataset=data_set, batch_size=5)

# train the model using 100 epochs of mini-batch gradient descent
LOSS = []
epochs = 100
#
for epoch in range(epochs):
    for x,y in train_loader:
        #make a prediction 
        yhat=model(x)
        #calculate the loss
        loss=criterion(yhat,y)
        #store loss/cost 
        LOSS.append(loss.item())
        #clear gradient 
        optimizer.zero_grad()
        #Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        #the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    # for every 10 epochs, print the loss
    print(f"Epoch {epoch} Loss {loss.item():.4f}")
# plot the loss
plt.plot(LOSS)
plt.xlabel("Iterations ")
plt.ylabel("Loss")
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):               # Importing the base NN module

        def __init__(self):             # Defining the constructor
            super(Net, self).__init__()          # Calling the Constructor of the base class

            # Defining the NN Architecture
            self.conv1 = nn.Conv2d(3,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84) 
            self.fc3 = nn.Linear(84,10)

        def forward(self,x):             # Takes in the input as well
            #Implementing forward propagation
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 400)          #Reshapes the tensor to the linear layer
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
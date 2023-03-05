import torch.optim as optim

criterion = nn.CrossEntropyLoss()                    #Defining the Loss function
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.8)       #Stochastic Gradient Descent Optimization
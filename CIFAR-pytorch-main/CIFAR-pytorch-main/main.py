import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from src.data_processing import Dataloading
from src.model import Net
from src.train import train_model


if __name__ == '__main__':

    # Checking if GPU is available:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    #Loading Dataset
    trainloader , testloader = Dataloading(batch = 4)

    #Create a model
    net = Net().to(device)
    print(net)

    #Train Model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    num_epochs = 10

    model, train_losses, train_accs, test_losses, test_accs = train_model(
        net, device, trainloader, testloader, criterion, optimizer, num_epochs)

    # Save trained model
    torch.save(model, 'model.pth')
    train_accs = [train_x.cpu() for train_x in train_accs]
    test_accs = [test_x.cpu() for test_x in test_accs]
    
    # Save training and validation losses and accuracies
    with open('train_losses.txt', 'w') as f:
        for loss in train_losses:
            f.write(str(loss) + '\n')
    with open('train_accs.txt', 'w') as f:
        for acc in train_accs:
            f.write(str(acc) + '\n')
    with open('test_losses.txt', 'w') as f:
        for loss in test_losses:
            f.write(str(loss) + '\n')
    with open('test_accs.txt', 'w') as f:
        for acc in test_accs:
            f.write(str(acc) + '\n')
    
    # Plot training and validation losses and accuracies
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title('Training Loss')
    axs[0, 1].plot(train_accs)
    axs[0, 1].set_title('Training Accuracy')
    axs[1, 0].plot(test_losses)
    axs[1, 0].set_title('Validation Loss')
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title('Validation Accuracy')
    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='')
    plt.show()

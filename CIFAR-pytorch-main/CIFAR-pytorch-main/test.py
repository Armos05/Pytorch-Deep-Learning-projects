import torch
from src.data_processing import Dataloading
from src.model import CNN
from src.train import train_model

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # Load data
    _, test_dataloader = Dataloading(batch_size=64)

    # Load trained model
    model = torch.load('model.pth')

    # Evaluate model
    model.eval()
    running_loss = 0.0
   

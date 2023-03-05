import torch
import torchvision                            #Has Datasets
import torchvision.transforms as transforms   #Transforming Datasets


def Dataloading(batch):
#Applying data transformation (converting image to Tensor) and (Normalization)
    transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    #Loading the Training Data
    trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True,transform = transform)

    #Wrapping the dataset to make it easier to iterate and orgnaize it in batches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch, shuffle = True, num_workers = 2)

    #Loading the Testing Data
    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True,transform = transform)

    #Wrapping the dataset to make it easier to iterate and orgnaize it in batches
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch, shuffle = False, num_workers = 2)

    return trainloader, testloader

import torchvision

def loadData():
    train_data = torchvision.datasets.SVHN(
        root="./data", split='train', 
        download=True, 
        transform=torchvision.transforms.ToTensor()
    )

    test_data = torchvision.datasets.SVHN(
        root="./data", split='test', 
        download=True, 
        transform=torchvision.transforms.ToTensor()
    )
    
    return train_data, test_data




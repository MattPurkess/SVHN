import torchvision
import scipy

def loadData():
    train_data = torchvision.datasets.SVHN(
        root="./data", split='train', 
        download=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(size=32, padding=4),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            torchvision.transforms.Normalize(
                mean=[0.4377, 0.4438, 0.4728],
                std=[0.1980, 0.2010, 0.1970]
                ),
        ])
    )

    test_data = torchvision.datasets.SVHN(
        root="./data", split='test', 
        download=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.4377, 0.4438, 0.4728],
                std=[0.1980, 0.2010, 0.1970]
                ),
        ])
    )

    return train_data, test_data




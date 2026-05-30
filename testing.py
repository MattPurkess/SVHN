import torch
import torchvision
import torch.nn as nn

def testing(model, test_data):
    total = 0

    testing_loader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle=False)

    for images, labels in testing_loader:
        predictions = model(images.to('cuda')).argmax(dim=1)

        total += (predictions == labels.to('cuda')).sum()

    print (total.item(), "/", len(test_data))
    print (total.item()/len(test_data), "%")
    
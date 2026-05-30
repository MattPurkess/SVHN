import torch
import torch.nn as nn

def training(model, data):
    epochs = 10

    batches = torch.utils.data.DataLoader(dataset=data, batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(params = model.parameters(), lr = 0.1)

    for i in range (epochs):
        for images, labels in batches:
            images = images.to('cuda')
            labels = labels.to('cuda')
            predictions = model(images)
            optimiser.zero_grad()
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimiser.step()
        print("Epoch:", i+1)
    
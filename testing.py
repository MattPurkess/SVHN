import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def testing(model, test_data):
    total = 0

    testing_loader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle=False)

    for images, labels in testing_loader:
        predictions = model(images.to('cuda')).argmax(dim=1)
        total += (predictions == labels.to('cuda')).sum()

    print (total.item(), "/", len(test_data))
    print (f"{100*total.item()/len(test_data):.2f}%")

    fig, axes = plt.subplots(1, 10, figsize=(20,2))
    for i in range (10):
        image, label = test_data[i]
        prediction = model(image.unsqueeze(0).to('cuda')).argmax(dim=1)
        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].set_title(f"Guess:{prediction.item()},Label:{label}", fontsize = 7)
    plt.show()
    
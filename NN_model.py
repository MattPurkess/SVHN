import torch
import torch.nn as nn

class MNIST_Model (nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.p1 = nn.MaxPool2d(2)
        self.C2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.p2 = nn.MaxPool2d(2)
        self.L1 = nn.Linear(32*8*8, 128)
        self.A1 = nn.ReLU()
        self.L2 = nn.Linear(128, 10)
        
    def forward(self, input):
        
        c1 = self.C1(input)
        r1 = self.A1(c1)
        p1 = self.p1(r1)
        c2 = self.C2(p1)
        r2 = self.A1(c2)
        p2 = self.p2(r2)

        x0 = torch.flatten(p2, 1)
        y1 = self.L1(x0)
        x1 = self.A1(y1)
        y2 = self.L2(x1)

        return y2








        
    
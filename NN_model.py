import torch
import torch.nn as nn

class SVHN_Model (nn.Module):
    def __init__(self):
        super().__init__()
        
        self.C1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.N1 = nn.BatchNorm2d(16)
        self.P1 = nn.MaxPool2d(2)
        self.C2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.N2 = nn.BatchNorm2d(32)
        self.P2 = nn.MaxPool2d(2)
        self.L1 = nn.Linear(32*8*8, 128)
        self.A1 = nn.ReLU()
        self.L2 = nn.Linear(128, 10)
        
    def forward(self, input):
        
        c1 = self.C1(input)
        n1 = self.N1(c1)
        r1 = self.A1(n1)
        p1 = self.P1(r1)
        c2 = self.C2(p1)
        n2 = self.N2(c2)
        r2 = self.A1(n2)
        p2 = self.P2(r2)

        x0 = torch.flatten(p2, 1)
        y1 = self.L1(x0)
        x1 = self.A1(y1)
        y2 = self.L2(x1)

        return y2








        
    
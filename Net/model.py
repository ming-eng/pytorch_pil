from torch import nn
from torch.autograd import Variable
from torch.nn import Module, Linear, Sequential, Conv2d, ReLU, ConstantPad2d
import torch.nn.functional as F


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnnLayers = Sequential(
            # padding添加1层常数1,设定卷积核为2*2
            ConstantPad2d(1, 1),
            Conv2d(1, 1, kernel_size=2, stride=2, bias=True)
        )
        self.linearLayers = Sequential(
            Linear(9, 2)
        )

    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.shape[0], -1)
        x = self.linearLayers(x)
        return x


class Net2(Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.cnnLayers = Sequential(
            Conv2d(1, 1, kernel_size=1, stride=1, bias=True)
        )
        self.linearLayers = Sequential(
            ReLU(),
            Linear(16, 2)
        )

    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.shape[0], -1)
        x = self.linearLayers(x)
        return x
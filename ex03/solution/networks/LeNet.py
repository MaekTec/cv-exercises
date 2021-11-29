import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # define functions here
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Todo
        pass


    def forward(self, x):
        # define forward pass here. Note that the functions from torch.nn.functional can be called directly here and must
        # not be defined in __init__()
        # Todo
        pass


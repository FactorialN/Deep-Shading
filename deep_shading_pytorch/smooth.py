import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def laplacian_kernerl(channel):
    
    laplacian_kernerl = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
    laplacian_kernerl = laplacian_kernerl.clone().repeat(channel,1,1,1)

    return Variable(laplacian_kernerl.cuda())

class Laplacian_warp(torch.nn.Module):
    def __init__(self, channel):
        super(Laplacian_warp, self).__init__()
        self.channel = channel
        self.kernerl = laplacian_kernerl(channel)

    def forward(self, input):

        return F.conv2d(input, self.kernerl, groups = self.channel)
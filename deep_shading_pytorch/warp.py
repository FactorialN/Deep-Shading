import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def create_meshgrid(batch_size, height, width):
	x = np.linspace(0, width, width) #width
	y = np.linspace(0, height, height) #height
	[X, Y] = np.meshgrid(x, y)
	X = X[:, :, np.newaxis]
	Y = Y[:, :, np.newaxis]
	meshgrid = np.concatenate((X, Y), axis=2)
	meshgrid = torch.Tensor(meshgrid)
	meshgrid = meshgrid.unsqueeze(0)
	meshgrid = meshgrid.clone().repeat(batch_size,1,1,1)

	return Variable(meshgrid.cuda())

class Warp(torch.nn.Module):
    def __init__(self, batch_size, height, width):
        super(Warp, self).__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.meshgrid = create_meshgrid(self.batch_size,self.height, self.width)

    def forward(self, img1, img2, flo):
    	ones = torch.ones_like(img1)
    	grid_0 = (self.meshgrid[:,:,:,0] + flo[:,:,:,0]).unsqueeze(3) / self.width * 2 - 1
    	grid_1 = (self.meshgrid[:,:,:,1] + flo[:,:,:,1]).unsqueeze(3) / self.height * 2 - 1
    	grid = torch.cat((grid_0, grid_1), 3)

    	mask = F.grid_sample(ones, grid)

    	return img1 * mask, F.grid_sample(img2, grid) * mask
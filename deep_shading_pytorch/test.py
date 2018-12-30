import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np
from smooth import Laplacian_warp
import pyexr



img = pyexr.open('0000000000.exr')
img1 = torch.from_numpy(np.rollaxis(img.get("diffuse"), 2)).float().unsqueeze(0)
img1 = Variable( img1.cuda(),  requires_grad=False)
laplacian_warp = Laplacian_warp(3)
res = laplacian_warp(img1)

print(res.data.shape)
res_tensor = res.data.cpu()[0]
res_tensor = res_tensor[0].unsqueeze(2).abs() + res_tensor[1].unsqueeze(2).abs() + res_tensor[2].unsqueeze(2).abs()
#res_tensor = torch.cat([res_tensor[0].unsqueeze(2), res_tensor[1].unsqueeze(2), res_tensor[2].unsqueeze(2)],2)
#res_tensor = res_tensor.abs()
#res_tensor = res_tensor < 0.03
res_pre = res_tensor.numpy()


img = pyexr.open('0000000000.exr')
img1 = torch.from_numpy(np.rollaxis(img.get("normal"), 2)).float().unsqueeze(0)
img1 = Variable( img1.cuda(),  requires_grad=False)
laplacian_warp = Laplacian_warp(3)
res = laplacian_warp(img1)

print(res.data.shape)
res_tensor = res.data.cpu()[0]
res_tensor = res_tensor[0].unsqueeze(2).abs() + res_tensor[1].unsqueeze(2).abs() + res_tensor[2].unsqueeze(2).abs()
#res_tensor = torch.cat([res_tensor[0].unsqueeze(2), res_tensor[1].unsqueeze(2), res_tensor[2].unsqueeze(2)],2)
#res_tensor = res_tensor.abs()
res_tensor = res_tensor < 0.0003
res_nor = res_tensor.numpy()

#res_tensor = np.abs(res_tensor)
pyexr.write("res.exr", res_pre   * res_nor)
#cv2.imwrite("res.png", np.reshape(res.data.cpu().numpy()[0][0], (254,254,1)))
#print(res.data)
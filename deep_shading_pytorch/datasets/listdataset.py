import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import pyexr
import numpy as np


def get_flow(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print ('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0], 2))
            #data2D = np.transpose(data2D,[0, 3,1,2])
            return data2D

def staircase_loader(root, path, next_path, path_flow):
    input = []
    image1 = pyexr.open(path)
    input.append(image1.get("diffuse_dir"))
    input.append(image1.get("normal"))
    #input.append(((image1.get("depth") - 0.5) * 5 + 1) / 2)
    input.append(image1.get("depth"))
    input.append(image1.get("albedo"))

    next_input = []
    image2 = pyexr.open(next_path)
    next_input.append(image2.get("diffuse_dir"))
    next_input.append(image2.get("normal"))
    #next_input.append(((image2.get("depth") - 0.5) * 5 + 1) / 2)
    next_input.append(image2.get("depth"))
    next_input.append(image2.get("albedo"))

    return input, image1.get("diffuse"), next_input, image2.get("diffuse"), get_flow(path_flow)


def default_loader(root, path_imgs, path_gt, path_next_imgs, path_next_gt, path_flow):
    input = []
    for i, img in enumerate(path_imgs):
        if i == 0:
            input.append(pyexr.read(img)[:,:,:3])
        else:
            input.append(pyexr.read(img)[:,:,:3])

    next_input = []

    for i, img in enumerate(path_next_imgs):
        if i == 0:
            next_input.append(pyexr.read(img)[:,:,:3])
        else:
            next_input.append(pyexr.read(img)[:,:,:3])

    return input, pyexr.read(path_gt)[:,:,:3], next_input, pyexr.read(path_next_gt)[:,:,:3], get_flow(path_flow)


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=staircase_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        image_1, image_2, flow_map = self.path_list[index]

        inputs, target, next_inputs, next_target, flow_map = self.loader(self.root, image_1, image_2, flow_map)

        #inputs, target, next_inputs, next_target, flow_map  = self.path_list[index]

        #inputs, target, next_inputs, next_target, flow_map = self.loader(self.root, inputs, target, next_inputs, next_target, flow_map)
        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
            next_inputs, next_target = self.co_transform(next_inputs, next_target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
            inputs[2] = self.transform(inputs[2])
            inputs[3] = self.transform(inputs[3])

            next_inputs[0] = self.transform(next_inputs[0])
            next_inputs[1] = self.transform(next_inputs[1])
            next_inputs[2] = self.transform(next_inputs[2])
            next_inputs[3] = self.transform(next_inputs[3])
        if self.target_transform is not None:
            target = self.target_transform(target)
            next_target = self.target_transform(next_target)

        return inputs, target, next_inputs, next_target, flow_map

    def __len__(self):
        return len(self.path_list)

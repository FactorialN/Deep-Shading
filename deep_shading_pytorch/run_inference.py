import argparse
from path import Path

import pyexr
import torch
import torch.backends.cudnn as cudnn
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
import glob
import os
import pytorch_ssim

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch Deep_video inference on a folder',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to images folder which must contain albedo, diffuse, normal, position')
parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')



def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img in '{}'".format(args.data))
    save_path = data_dir/'predict'
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()


    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])

    dir = args.data
    images = []

    for image in sorted(glob.glob(os.path.join(dir, '*.exr'))):
        if image[-5] == 'a':
            continue

        if len(images) < 10000:
            images.append([image])

    # for image in sorted(glob.glob(os.path.join(dir, 'Diffuse/*.exr'))):
    #     if len(images) < 10000000:
    #         basename = image[-14:]
    #         diffuse = os.path.join(dir, 'Diffuse', basename)
    #         normal = os.path.join(dir, 'Normal', basename)
    #         position = os.path.join(dir, 'Position', basename)
    #         gt = os.path.join(dir, 'GT', basename)
    #         direct = os.path.join(dir, 'Direct', basename)
    #         albedo = os.path.join(dir, 'Albedo', basename)
            
    #         images.append([diffuse, normal, position, gt, direct, albedo])

    print('{} samples found'.format(len(images)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).cuda()
    model.eval()
    cudnn.benchmark = True

    l1loss = torch.nn.L1Loss()

    for image in images:
        image_exr = pyexr.open(image[0])
        diffuse_0 = input_transform(image_exr.get("diffuse_dir")[:,:,0][:,:,np.newaxis])
        diffuse_1 = input_transform(image_exr.get("diffuse_dir")[:,:,1][:,:,np.newaxis])
        diffuse_2 = input_transform(image_exr.get("diffuse_dir")[:,:,2][:,:,np.newaxis])
        normal_single = input_transform(image_exr.get("normal"))
        #position_single = input_transform((image_exr.get("depth") - 0.5) * 0.5 + 0.5)

        position_single = input_transform((image_exr.get("depth")))

        gt_0 = input_transform(image_exr.get("diffuse")[:,:,0][:,:,np.newaxis])

        gt_0 = gt_0.unsqueeze(0)

        #diffuse_0 = input_transform(pyexr.read(image[0])[:,:,0][:,:,np.newaxis])
        #diffuse_1 = input_transform(pyexr.read(image[0])[:,:,1][:,:,np.newaxis])
        #diffuse_2 = input_transform(pyexr.read(image[0])[:,:,2][:,:,np.newaxis])
        #normal_single = input_transform(pyexr.read(image[1])[:,:,:3])
        #position_single = input_transform(pyexr.read(image[2])[:,:,:3])
        
        #gt_0 = input_transform(pyexr.read(image[3])[:,:,0][:,:,np.newaxis])
        #gt_0 = gt_0.unsqueeze(0)

        diffuse_0 = diffuse_0.unsqueeze(0)
        diffuse_1 = diffuse_1.unsqueeze(0)
        diffuse_2 = diffuse_2.unsqueeze(0)

        position = position_single.unsqueeze(0)
        normal = normal_single.unsqueeze(0)

        # normal = torch.rand(3,normal_single.shape[0],normal_single.shape[1], normal_single.shape[2])
        # normal[0] = normal_single
        # normal[1] = normal_single
        # normal[2] = normal_single

        # position = torch.rand(3,position_single.shape[0],position_single.shape[1], position_single.shape[2])
        # position[0] = position_single
        # position[1] = position_single
        # position[2] = position_single

        input_0 = torch.cat([diffuse_0, normal, position], 1)
        input_1 = torch.cat([diffuse_1, normal, position], 1)
        input_2 = torch.cat([diffuse_2, normal, position], 1)

        input_var_0 = torch.autograd.Variable(input_0.cuda(), volatile=True)
        input_var_1 = torch.autograd.Variable(input_1.cuda(), volatile=True)
        input_var_2 = torch.autograd.Variable(input_2.cuda(), volatile=True)
        #input = torch.cat([input_0, input_1, input_2], 0)

        #input_var = torch.autograd.Variable(input.cuda(), volatile=True)

        output_0 = model(input_var_0)
        output_1 = model(input_var_1)
        output_2 = model(input_var_2)

        #loss = l1loss(torch.autograd.Variable(gt_0.cuda(), volatile=True), output_0)
        #print (loss.data)

        output_tensor_0 = output_0.data.view(300,400,1)
        output_tensor_1 = output_1.data.view(300,400,1)
        output_tensor_2 = output_2.data.view(300,400,1)

        output_tensor = torch.cat([output_tensor_0, output_tensor_1, output_tensor_2], 2)
        #print (output_tensor.shape)
        
        ssim_loss = pytorch_ssim.SSIM()
        output_numpy = output_tensor.cpu().numpy()
        output_numpy = output_tensor.cpu().numpy() * image_exr.get("albedo") + image_exr.get("diffuse_dir") + image_exr.get("specular_dir")
        
        #output_numpy[:,:,1] = image_exr.get("diffuse")[:,:,1]
        #output_numpy[:,:,2] = image_exr.get("diffuse")[:,:,2]

        #output_numpy = output_numpy[:,:,0][:,:,np.newaxis]
        #output_numpy = np.repeat(output_numpy, 3, axis = 2)
        
        print(image[0][-14:])
        #print(output_numpy.shape)

        loss_channel = []

        for i in [0, 1, 2]:
            im1 = output_numpy[:,:,i][np.newaxis,np.newaxis,:,:]
            #print (type(im1))
            im1 = torch.autograd.Variable(torch.from_numpy(im1))
            #print(im1.size())
            im2 = image_exr.get("diffuse")[:,:,i][np.newaxis,np.newaxis,:,:]
            #print(im2.size())
            im2 = torch.autograd.Variable(torch.from_numpy(im2))

            diff = im1 - im2
            #loss_channel.append(l1loss(diff, torch.autograd.Variable(torch.zeros_like(diff.data)) ).data)
            loss_channel.append(ssim_loss(im1, im2).data)

        print (loss_channel)

        #output_numpy = image_exr.get("diffuse_dir") 

        #output_numpy  = output_tensor.cpu().numpy() * image_exr.get("albedo") + image_exr.get("diffuse_dir") + image_exr.get("specular_dir")
        #output_numpy = output_tensor.view(output_tensor.shape[2], output_tensor.shape[3], output_tensor.shape[0]).cpu().numpy()

        #print (output_numpy)
        #print (np.sum(output_numpy > 0), 256 * 256 * 3)
        #pyexr.write(save_path / image[0][-14:], pyexr.read(image[4])[:,:,:3]  + pyexr.read(image[5])[:,:,:3] * np.absolute(output_numpy))
        pyexr.write(save_path / image[0][-14:], np.absolute(output_numpy))





if __name__ == '__main__':
    main()

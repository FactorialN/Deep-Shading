import argparse
import os
import shutil
import time

import datetime
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import pytorch_ssim
import flow_transforms
import models
import datasets
import pytorch_ssim

from warp import Warp
from smooth import Laplacian_warp

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='deep_vedio',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=1.0, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')
parser.add_argument('--arch', '-a', metavar='ARCH', default='unet',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=30000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--milestones', default=[1000,3000,5000, 10000, 20000, 40000, 80000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

best_EPE = -1
n_iter = 0


def main():
    global args, best_EPE, save_path
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    #test_writer = SummaryWriter(os.path.join(save_path,'test'))
    #output_writers = []
    #for i in range(3):
    #    output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])

    # if 'KITTI' in args.dataset:
    #     args.sparse = True
    # if args.sparse:
    #     co_transform = flow_transforms.Compose([
    #         flow_transforms.RandomCrop((320,448)),
    #         flow_transforms.RandomVerticalFlip(),
    #         flow_transforms.RandomHorizontalFlip()
    #     ])
    # else:
    #     co_transform = flow_transforms.Compose([
    #         flow_transforms.RandomTranslate(10),
    #         flow_transforms.RandomRotate(10,5),
    #         flow_transforms.RandomCrop((320,448)),
    #         flow_transforms.RandomVerticalFlip(),
    #         flow_transforms.RandomHorizontalFlip()
    #     ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                    len(test_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    #optimizer = torch.optim.Adadelta(param_groups,1.0,0.9)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, train_writer)
        #train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # evaluate on validation set

        #EPE = validate(val_loader, model, epoch)
        #test_writer.add_scalar('mean EPE', EPE, epoch)

        # if best_EPE < 0:
        #     best_EPE = EPE

        # is_best = EPE < best_EPE
        # best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict()
        }, False)


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()
    ssim_loss = pytorch_ssim.SSIM()
    l1loss = torch.nn.L1Loss()

    end = time.time()

    for i, (input, target, next_input, next_target, flow_map) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target.cuda(async=True)
        
        warp = Warp(flow_map.shape[0], flow_map.shape[1], flow_map.shape[2])
        l1_warp = Laplacian_warp(1)
        l3_warp = Laplacian_warp(3)

        diffuse = input[0]
        normal = input[1]
        position = input[2]
        albedo = input[3].cuda()
        dark_weight = torch.clamp((1.0 - diffuse) / torch.max(1.0 - diffuse), min = 0, max = 1.0)
        #dark_weight = dark_weight / (1.001 - dark_weight)
        dark_weight = torch.ones_like(diffuse)


        next_diffuse = next_input[0]
        next_normal = next_input[1]
        next_position = next_input[2]
        next_albedo = next_input[3].cuda()
        next_dark_weight = torch.clamp((1.0 - next_diffuse) / torch.max(1.0 - next_diffuse), min = 0, max = 1.0)
        #next_dark_weight = next_dark_weight / (1.001 - next_dark_weight)
        #next_dark_weight = torch.ones_like(next_diffuse)

        normal_var = torch.autograd.Variable(normal.cuda())
        next_normal_var = torch.autograd.Variable(next_normal.cuda())

        l_normal_var = l3_warp(normal_var).abs()
        l_normal_var = (l_normal_var[:,0,:,:].unsqueeze(1) + l_normal_var[:,1,:,:].unsqueeze(1) + l_normal_var[:,2,:,:].unsqueeze(1)) < 0.0003
        l_normal_var = l_normal_var.float()

        l_next_normal_var = l3_warp(next_normal_var).abs()
        l_next_normal_var = (l_next_normal_var[:,0,:,:].unsqueeze(1) + l_next_normal_var[:,1,:,:].unsqueeze(1) + l_next_normal_var[:,2,:,:].unsqueeze(1)) < 0.0003
        l_next_normal_var = l_next_normal_var.float()

        flow_var = torch.autograd.Variable(flow_map.cuda())

        #chennel_rand = np.floor(np.random.rand(1)[0] * 3).astype(int)
        #chennel_rand = epoch % 3
        chennel_rand = np.arange(3)
        #np.random.shuffle(chennel_rand)

        

        for channel in chennel_rand:
            rand_scale = np.random.rand(1)[0] * 1.5 + 0.5
            dark_weight_var = torch.autograd.Variable(dark_weight[:,channel,:,:].unsqueeze(1).cuda())
            next_dark_weight_var = torch.autograd.Variable(next_dark_weight[:,channel,:,:].unsqueeze(1).cuda())


            input = [diffuse[:,channel,:,:].unsqueeze(1) * rand_scale, normal, position]
            input = [j.cuda() for j in input]
            input_var = torch.autograd.Variable(torch.cat(input,1))
            target_var = torch.autograd.Variable(target[:,channel,:,:].unsqueeze(1).cuda() * rand_scale)
            albedo_var = albedo[:,channel,:,:].unsqueeze(1)
            albedo_var = torch.autograd.Variable(albedo_var.cuda())

            next_input = [next_diffuse[:,channel,:,:].unsqueeze(1) * rand_scale, next_normal, next_position]
            next_input = [j.cuda() for j in next_input]
            next_input_var = torch.autograd.Variable(torch.cat(next_input,1))
            next_target_var = torch.autograd.Variable(next_target[:,channel,:,:].unsqueeze(1).cuda() * rand_scale)
            next_albedo_var = next_albedo[:,channel,:,:].unsqueeze(1)
            next_albedo_var = torch.autograd.Variable(next_albedo_var.cuda())

            # compute output
            output = model(input_var)
            next_output = model(next_input_var)
            #print output.shape, next_output.shape
            warp_output1, warp_output2 = warp(output, next_output, flow_var)

            l_output = l1_warp(output)
            l_next_output = l1_warp(next_output)

            l_target_var = l1_warp(target_var / (albedo_var + 0.01))
            l_next_target_var = l1_warp(next_target_var / (next_albedo_var + 0.01))
            #loss = l1loss(output, target_var)
            
            shading_loss = ssim_loss(output * albedo_var , target_var)
            next_shading_loss = ssim_loss(next_output * next_albedo_var , next_target_var )

            #diff = target_var - output * albedo_var
            #next_diff = next_target_var - next_output * next_albedo_var
            
            shading_l1_loss = l1loss(output * albedo_var * dark_weight_var, target_var * dark_weight_var)
            next_shading_l1_loss = l1loss(next_output * next_albedo_var * next_dark_weight_var, next_target_var * next_dark_weight_var)


            #smooth_loss = l1loss(l_output * l_normal_var, torch.autograd.Variable(torch.zeros_like(l_output.data)))
            #next_smooth_loss = l1loss(l_next_output * l_next_normal_var, torch.autograd.Variable(torch.zeros_like(l_next_output.data)))

            smooth_loss = l1loss(l_output * l_normal_var, l_target_var * l_normal_var)
            next_smooth_loss = l1loss(l_next_output * l_next_normal_var, l_next_target_var * l_next_normal_var)
            #temporal_loss = ssim_loss(warp_output1, warp_output2)
            #print warp_output1.requires_grad
            #print warp_output2.requires_grad
            warp_res = warp_output2 - warp_output1
            temporal_loss = l1loss(warp_res, torch.autograd.Variable(torch.zeros_like(warp_res.data)))

            #loss = shading_loss
            #print "##########"
            #print (shading_loss + next_shading_loss) , 10 * temporal_loss
            weight = 50
            if epoch > 1000:
                weight = 40
            if epoch > 2500:
                weight = 80

            #loss = 2 * shading_loss + 2 * next_shading_loss + shading_l1_loss * 0.5 + next_shading_l1_loss * 0.5 + weight * temporal_loss
            loss = shading_loss + next_shading_loss + shading_l1_loss * 5 + next_shading_l1_loss * 5 + smooth_loss * 10 + next_smooth_loss * 30 + weight * temporal_loss
            # record loss and EPE
            losses.update(loss.data[0], target.size(0))
            train_writer.add_scalar('train_loss', loss.data[0], n_iter)
            train_writer.add_scalar('temporal_loss', temporal_loss.data[0], n_iter)
            train_writer.add_scalar('l1_loss', shading_l1_loss.data[0], n_iter)
            train_writer.add_scalar('ssim_loss', shading_loss.data[0], n_iter)
            train_writer.add_scalar('smooth_loss', smooth_loss.data[0], n_iter)
            n_iter += 1

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses))
        if i >= epoch_size:
            break

    return losses.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(torch.cat(input,1).cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        flow2_EPE = args.div_flow*realEPE(output, target_var, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.data[0], target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):  # log first output of first batches
            if epoch == 0:
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0].cpu().numpy(), max_value=10), 0)
                output_writers[i].add_image('Inputs', input[0][0].numpy().transpose(1, 2, 0) + np.array([0.411,0.432,0.45]), 0)
                output_writers[i].add_image('Inputs', input[1][0].numpy().transpose(1, 2, 0) + np.array([0.411,0.432,0.45]), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output.data[0].cpu().numpy(), max_value=10), epoch)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


if __name__ == '__main__':
    main()

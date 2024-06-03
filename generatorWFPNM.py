from __future__ import print_function
import sys
import os
sys.path.append('os.path.dirname(os.path.realpath(__file__))')
import datetime
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import WGANFPNX
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--imsize', type=int, default=1, help='the height of the z tensor')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--weight_clip_value', type=float, default=0.01, help='weight clip value')
parser.add_argument('--gradient_penalty_weight', type=float, default=5, help='gradient_penalty')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')

opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

n_levels = 4  # 你需要根据你的需求来设置这个值
overall_start_time = datetime.datetime.now()
# 循环从43到64的种子值
for seed in range(43, 44):
    # 在每个循环迭代开始时记录时间
    start_time = datetime.datetime.now()
    opt.seed = seed
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    print("begin init netG for seed", seed)
    netG = WGANFPNX.LaplacianPyramidGenerator(opt.imageSize, nz, nc, ngf, ngpu)
    print("netG init end.")
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print('netG is:\n')
    print(netG)
    print(dir(netG.__class__))

    fixed_noise = torch.FloatTensor(1, nz, opt.imsize, opt.imsize, opt.imsize).normal_(0, 1)
    if opt.cuda:
        netG.cuda()
        fixed_noise = fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    print('Generating for seed', seed, '...')
    for level in range(n_levels):
        if level == 0:
            fake = netG(fixed_noise, level)
        else:
            fake = netG(fixed_noise, level, fake)

        torch.cuda.empty_cache()
        # 在每次迭代结束后清除GPU缓存
        torch.cuda.empty_cache()

    print('Saving for seed', seed, '...')
    save_hdf5(fake.data, 'XXSAV{1}_{2}.hdf5'.format(opt.experiment, opt.experiment, seed))
    print('Saved to', opt.experiment, 'folder for seed', seed)

    # 清除GPU缓存和删除变量
    del netG
    del fake
    torch.cuda.empty_cache()

    print('Memory cleared for seed', seed)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for seed {seed}: {elapsed_time}")

overall_end_time = datetime.datetime.now()
overall_elapsed_time = overall_end_time - overall_start_time
print(f"Overall elapsed time for all seeds: {overall_elapsed_time}")
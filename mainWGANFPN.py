from __future__ import print_function

# coding=utf-8
#!/usr/bin/env python3
import sys
import os
sys.path.append('os.path.dirname(os.path.realpath(__file__))')


import sys
import os
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataset import HDF5Dataset
from hdf5_io import save_hdf5
import WGANFPNX
import numpy as np
import time


def main():
    time_start = time.time()
    torch.cuda.set_device(0)
    print('start time is', time_start)
    np.random.seed(43)

    # Change workdir to where you want the files output
    work_dir = os.path.expandvars('G:/multiphase/multi/')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='3D')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--weight_clip_value', type=float, default=0.01, help='weight clip value')
    parser.add_argument('--gradient_penalty_weight', type=float, default=5, help='gradient_penalty')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    opt.manualSeed = 43  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.dataset in ['3D']:
        dataset = HDF5Dataset(opt.dataroot,
                              input_transform=transforms.Compose([
                                  transforms.ToTensor()
                              ]))
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers),
                                             pin_memory=True)

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    weight_decay = opt.weight_decay
    weight_clip_value = opt.weight_clip_value
    weight_gradient_penalty_weight = opt.gradient_penalty_weight
    nc = 1

    # Custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    device = torch.device("cuda" if opt.cuda else "cpu")
    n_levels = 4  
    netG = WGANFPNX.LaplacianPyramidGenerator(opt.imageSize, nz, nc, ngf, ngpu).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = WGANFPNX.WGAN3D_D(opt.imageSize, nz, nc, ndf, ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    wasserstein_criterion = WGANFPNX.WassersteinLP(opt.weight_clip_value, weight_gradient_penalty_weight, netD)

    input, noise, fixed_noise, fixed_noise_TI = None, None, None, None
    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize, opt.imageSize).to(device)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1, 1).normal_(0, 1).to(device)
    fixed_noise = torch.FloatTensor(1, nz, 7, 7, 7).normal_(0, 1).to(device)
    fixed_noise_TI = torch.FloatTensor(1, nz, 1, 1, 1).normal_(0, 1).to(device)

    label = torch.FloatTensor(opt.batchSize).to(device)
    real_label = 1
    fake_label = 0

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)

    fixed_noise = Variable(fixed_noise)
    fixed_noise_TI = Variable(fixed_noise_TI)

    # Setup optimizer
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    gen_iterations = 0
    dis_iterations = 0
    f = open(work_dir + "training_curve.csv", "a")
    for epoch in range(opt.niter):

        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with real
            netD.zero_grad()

            real_cpu = data.to(device)

            batch_size = real_cpu.size(0)
            #print(batch_size, gen_iterations, dis_iterations)

            input.resize_(real_cpu.size()).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)

            errD_real = netD(input)

            # Train with fake
            if dis_iterations % 10 == 0:
             netD.zero_grad()
             noise.resize_(batch_size, nz, 1, 1, 1)
             noise.data.normal_(0, 1)
             for level in range(n_levels):
                if level == 0:
                    fake = netG(noise, level)
                else:
                    fake = netG(noise, level, fake)
             fake = fake.cpu()  # Move the generated fake samples to CPU
             fake = fake.cuda()  # Move generated fake samples back to GPU
             fake = fake.detach()

             #print("Shape of real_cpu:", real_cpu.shape)
             #print("Data type of real_cpu:", real_cpu.dtype)
             #print("Min and Max values of real_cpu:", torch.min(real_cpu).item(), torch.max(real_cpu).item())

             #print("Shape of fake:", fake.shape)
             #print("Data type of fake:", fake.dtype)
             #print("Min and Max values of fake:", torch.min(fake).item(), torch.max(fake).item())

             errD_fake = netD(fake)
             errD = wasserstein_criterion(errD_real, errD_fake, real_cpu,fake)

             errD.backward()
             optimizerD.step()
            dis_iterations += 1
            #for p in netD.parameters():
                     #p.data.clamp_(-weight_clip_value, weight_clip_value)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if gen_iterations % 1 == 0:
                g_iter = 1
                while g_iter != 0:
                    netG.zero_grad()
                    label.data.fill_(1.0)  # Fake labels are real for generator cost
                    noise.data.normal_(0, 1)
                    for level in range(n_levels):
                        if level == 0:
                            fake = netG(noise, level)
                        else:
                            fake = netG(noise, level, fake)
                    pred_fake = netD(fake)
                    errG = -torch.mean(pred_fake)
                    errG.backward()
                    optimizerG.step()
                    g_iter -= 1
                gen_iterations += 1
            else:
                gen_iterations += 1

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, opt.niter, i, len(dataloader), errD.data, errG.data))
            f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                    % (epoch, opt.niter, i, len(dataloader), errD.data, errG.data))
            f.write('\n')

        if (epoch + 1) % 1 == 0:
            for level in range(n_levels):
                if level == 0:
                    fake = netG(fixed_noise, level)
                else:
                    fake = netG(fixed_noise, level, fake)
            save_hdf5(fake.data, work_dir + 'fake_samples_train_{0}_ppp.hdf5'.format(1 + epoch))

        # Do checkpointing
        if (epoch + 1) % 1 == 0:
            torch.save(netG.state_dict(), work_dir + 'netG_epoch_train_%d_ppp.pth' % (1 + epoch))
            torch.save(netD.state_dict(), work_dir + 'netD_epoch_train_%d_ppp.pth' % (1 + epoch))
    f.close()
    time_end = time.time()
    print('end time is', time_end)
    print('total time cost is', time_end - time_start)


if __name__ == "__main__":
    main()



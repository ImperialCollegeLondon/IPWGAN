# coding=utf-8
#!/usr/bin/env python3
import sys
import os
sys.path.append('os.path.dirname(os.path.realpath(__file__))')

import tifffile
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='path to image',default='imagecut.tif')
parser.add_argument('--name', required=True, help='name of dataset',default='berea')
parser.add_argument('--edgelengthXY', type=int, default=64, help='input batch size X and Y')
parser.add_argument('--edgelengthZ', type=int, default=32, help='input batch size Z')
parser.add_argument('--strideXY', type=int, default=32, help='the height / width of the input image to network, X and Y')
parser.add_argument('--strideZ', type=int, default=32, help='the height / width of the input image to network, Z')
parser.add_argument('--target_dir', required=True, help='path to store training images',default='dataset')

opt = parser.parse_args()
print(opt)

img = tifffile.imread(str(opt.image))

count = 0

edge_lengthXY = opt.edgelengthXY #image dimensions, X and Y
edge_lengthZ = opt.edgelengthZ #image dimensions, Z
strideXY = opt.strideXY #stride at which images are extracted, X and Y
strideZ = opt.strideZ #stride at which images are extracted, Z

N = edge_lengthXY
M = edge_lengthXY
O = edge_lengthZ

I_inc = strideXY
J_inc = strideXY
K_inc = strideZ

target_direc = str(opt.target_dir)
count = 0
for i in range(0, img.shape[0], I_inc):
    for j in range(0, img.shape[1], J_inc):
        for k in range(0, img.shape[2], K_inc):
            subset = img[i:i+N, j:j+M, k:k+O]
            if subset.shape == (N, M, O):
                f = h5py.File(target_direc+"/"+str(opt.name)+"_"+str(count)+".hdf5", "w")
                f.create_dataset('data', data=subset, dtype="i8", compression="gzip")
                f.close()
                count += 1
print(count)
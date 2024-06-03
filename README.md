# IPWGAN
Here is a Code of IPWGAN- Improved Pymarid Wasserstein Generative Adversarial Network

IPWGAN, Improved Pyramid Wasserstein Generative Adversarial Network, use to generate complex porous media.

## Features

- Feature 1: add pyramid structure generator to capture multiscale characterization
- Feature 2: add Feature Statistics Mixing Regularization to control the process of training
- Feature 3: add -LP to control training process.

## Usage

Create dataset: python create_training_imagess.py --image D:\fw3L.tif --name fw3r --target_dir I:\wet255128

Training: python mainWGANFPN.py --dataset 3D --dataroot SAV --imageSize 128 --batchSize 8 --ngf 64 --ndf 16 --nz 512 --niter 1600 --lr 0.00005 --ngpu 1 –cuda

Generation: python generatorWFPNM.py --imageSize 128 --ngf 64 --ndf 16 --nz 512 --netG SAVInetG_epoch_train_500_ppp.pth  --experiment nb --imsize 8 –cuda

## Paper

Cite as: Linqi Zhu, Branko Bijeljic, Martin Julian Blunt. Generation of Heterogeneous Pore-Space Images Using Improved Pyramid Wasserstein Generative Adversarial Networks. ESS Open Archive . December 03, 2023.
DOI: 10.22541/essoar.170158343.30188169/v1

 

#!/usr/bin/python3
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from net import deepIQA, weighted_loss, weights_init
import imageio
from utils import *
from dataset import *
from winlaic_utils import WinlaicLogger, Averager, ModelSaver
import pdb
import tqdm
from iqa_utils import LCC, SROCC
from matplotlib import pyplot as plt
import scipy
from PIL import Image

if __name__ == "__main__":
    net = torch.load('/home/cel-door/Codes/deepIQA_torch/saved_models/2019-03-21_23-43-30/2019-03-22_00-16-31_LCC:0.935/Model.mdl')
    # net.load_state_dict()
    net.eval().cuda()

    img = imageio.imread('/media/cel-door/6030688C30686B4C/winlaic_dataset/SCI/SIQAD/DistortedImages/cim2_2_7.bmp')
    patches = np.squeeze(extract_patches(img))
    quality_map = np.zeros(patches.shape[0:2])
    with torch.no_grad():
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                quality_map[i,j] = net(torch.tensor(patches[i,j,:,:,:], dtype=torch.float32).unsqueeze(0).transpose(1,3).transpose(2,3).float().cuda()).cpu().numpy()
    plt.imshow(quality_map, interpolation='bicubic', vmin=0, vmax=10)
    plt.colorbar()
    plt.figure()
    plt.imshow(img)
    plt.figure()
    grad = np.gradient(img[:,:,1])
    plt.imshow(grad[0])
    plt.show()
    pass
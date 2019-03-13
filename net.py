#!/usr/bin/python3
import torch, torchvision
import numpy as np
import torch.nn as NN
import torch.functional as F
import pdb

class VGGpart(NN.Module):
    def __init__(self,ch_in,ch_out,kernal_size=3, padding=1):
        super().__init__()
        self.conv = NN.Sequential(
            NN.Conv2d(ch_in,ch_out,kernal_size,padding=padding),
            NN.ReLU(inplace=True),
            NN.Conv2d(ch_out,ch_out,kernal_size,padding=padding),
            NN.ReLU(inplace=True),
            NN.MaxPool2d(2,2)
        )
    def forward(self,x):
        return self.conv(x)
        
class deepIQA(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv_b1 = NN.Sequential(
            VGGpart(3,32),VGGpart(32,64),VGGpart(64,128),
            VGGpart(128,256),VGGpart(256,512)
        )
        self.conv_b2 = NN.Sequential(
            VGGpart(3,32,5,2),VGGpart(32,64,5,2),VGGpart(64,128,5,2),
            VGGpart(128,256,5,2),VGGpart(256,512,5,2)
        )
        self.conv_b3 = NN.Sequential(
            VGGpart(3,32,7,3),VGGpart(32,64,7,3),VGGpart(64,128,7,3),
            VGGpart(128,256,7,3),VGGpart(256,512,7,3)
        )
        self.fc_weights = NN.Sequential(
            NN.Linear(512*3,512*3),NN.ReLU(),NN.Dropout(.5),
            NN.Linear(512*3,1),NN.ReLU()
        )
        self.fc_values = NN.Sequential(
            NN.Linear(512*3,512*3),NN.ReLU(),NN.Dropout(.5),
            NN.Linear(512*3,1),NN.ReLU()
        )
        # self.fc_weights = NN.Sequential(
        #     NN.Linear(1024,1024),NN.ReLU(),NN.Dropout(.5),
        #     NN.Linear(1024,1),NN.ReLU()
        # )
        # self.fc_values = NN.Sequential(
        #     NN.Linear(1024,1024),NN.ReLU(),NN.Dropout(.5),
        #     NN.Linear(1024,1),NN.ReLU()
        # )
    def forward(self,x):
        x1 = self.conv_b1(x)
        x2 = self.conv_b2(x)
        x3 = self.conv_b3(x)
        x = torch.cat([x1, x2, x3], 1)
        x = x.squeeze()
        a = self.fc_weights(x)
        a += 0.000001
        x = self.fc_values(x)
        
        return x, a

def weighted_loss(x, a, y, n_patch_per_img=32):
    scores = (a*x).reshape(-1,n_patch_per_img).sum(1)/a.reshape(-1,n_patch_per_img).sum(1)
    # pdb.set_trace()
    return (y-scores).abs().sum() # MAE
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        NN.init.normal_(m.weight.data, 0.0, 0.05)

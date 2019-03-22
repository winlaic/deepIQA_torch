#!/usr/bin/python3
import torch, torchvision
import numpy as np
import torch.nn as NN
import torch.nn.functional as F
import pdb

class ResPart(NN.Module):
    def __init__(self, ch_in, ch_out, kernal_size=3, padding=1, increase_dim_method = 'zero_padding'):
        super().__init__()
        self.conv=NN.Sequential(
            NN.Conv2d(ch_in, ch_out, kernal_size, padding=padding),
            NN.ReLU(inplace=True),
            NN.Conv2d(ch_out, ch_out, kernal_size, padding=padding)
        )
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.increase_dim_method = increase_dim_method
        if ch_in != ch_out and increase_dim_method == 'linear_transform':
            self.linear_channel_mapping = NN.Conv2d(ch_in, ch_out, 1)

    
    def forward(self, x):
        Fx = self.conv(x)
        if self.ch_in != self.ch_out:
            if self.increase_dim_method == 'zero_padding':
                x = Fx + F.pad(x, (0,0,0,0,0,self.ch_out-self.ch_in,0,0),'constant',0)
            elif self.increase_dim_method == 'linear_transform':
                x = Fx + self.linear_channel_mapping(x)
        else:
            x = Fx + x
        x = NN.functional.relu(x)
        return x


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
class DualPooling(NN.Module):
    def __init__(self, kernal_size, stride):
        super().__init__()
        self.max_pool = NN.MaxPool2d(kernal_size, stride)
        self.avg_pool = NN.AvgPool2d(kernal_size, stride)
    def forward(self, x):
        return torch.cat([self.max_pool(x), self.avg_pool(x)], 1)
  
class deepIQA(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv_b1 = NN.Sequential(
            NN.Conv2d(3, 50, 7),NN.ReLU(inplace=True),DualPooling(2, 2),
            NN.Conv2d(100, 100, 7),NN.ReLU(inplace=True),DualPooling(2,2)
        )
        self.dense = NN.Sequential(
            NN.Linear(3*3*200, 1024), NN.ReLU(),
            NN.Linear(1024, 1024), NN.ReLU(),
            NN.Linear(1024, 1), NN.ReLU()
        )

    def forward(self,x):
        x = self.conv_b1(x)
        x = x.view(-1,3*3*200)
        x = self.dense(x)
        return x

def weighted_loss(x, a, y, n_patch_per_img=32):
    scores = (a*x).reshape(-1,n_patch_per_img).sum(1)/a.reshape(-1,n_patch_per_img).sum(1)
    # pdb.set_trace()
    return (y-scores).abs().sum() # MAE
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        NN.init.normal_(m.weight.data, 0.0, 0.005)

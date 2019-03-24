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

from iqa_utils import LCC, SROCC
import os

run_on = os.environ['LOGNAME']
if run_on == 'lxye':
    data_base_dir = '/home/lxye/project/datasets'
    tqdm = lambda x:x
elif run_on == 'cel-door':
    data_base_dir = '/media/cel-door/6030688C30686B4C/winlaic_dataset'
    from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--data-dir',
                        default=os.path.join(data_base_dir,'SCI/SIQAD/DistortedImages'))
    parser.add_argument('-t', '--target-file',
                        default=os.path.join(data_base_dir,'SCI/SIQAD/DMOS.csv'))
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--save-freq', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5000, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('-j', '--num-threads', default=8)
    args = parser.parse_args()
    print(args)
    return args



def forward_data(net1, net2, x1, x2):
    x1 = net1(x1)
    x2 = net1(x2)
    y = torch.cat([x1, x2], 1)
    y = net2(y)
    return y


if __name__ == '__main__':
    args = parse_args()
    logger = WinlaicLogger()
    logger.w = ['Model','Try ResNet.']
    loss_collector = Averager()
    saver = ModelSaver()
    device = torch.device('cuda:0')
    # train_set, validation_set, test_set = generate_dataset(args.data_dir, args.target_file, part_ratio=(15,5,5))
    train_set, validation_set = generate_SIQAD(args.data_dir, args.target_file)
    net = deepIQA()
    net.cuda().train()
    net.apply(weights_init)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_threads,
                                             pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for _i in range(args.epochs):
        net.train()
        for patch, mos in tqdm(train_loader):
            patch = patch.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
            net.zero_grad()
            x = net(patch)
            x = x.view(-1, 32).mean(dim=1)
            y = mos.float().cuda()
            # y = torch.rand(4).cuda()
            loss = torch.nn.functional.mse_loss(x, y, size_average=True)
            loss.backward()
            loss_collector.add(loss.data)
            optimizer.step()
        
        logger.i=['epoch',_i ,'loss',loss_collector.mean]
        loss_collector.clear()
        if _i != 0 and _i % 10 == 0:
            net.eval()
            with torch.no_grad():
                y_all = torch.empty(size=(len(validation_set),),dtype=torch.float32)
                y_bar_all = torch.empty(size=(len(validation_set),),dtype=torch.float32)
                for i, (eval_img, eval_y) in enumerate(validation_set):
                    eval_img = eval_img.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
                    y_bar = net(eval_img).mean()                    
                    y_all[i] = torch.tensor(eval_y,dtype=torch.float32)
                    y_bar_all[i] = y_bar.cpu()
                this_lcc = LCC(y_all, y_bar_all).numpy()
                this_srocc = SROCC(y_all, y_bar_all).numpy()
                logger.w=['LCC', this_lcc]
                logger.w=['SROCC', this_srocc]
                if _i % args.save_freq == 0:
                    logger.w = ['Model saved']
                    save_dir = saver.save_dir(['LCC','%.3f' % float(this_lcc)])
                    torch.save(net, join(save_dir, 'Model.mdl'))
                    torch.save(optimizer, join(save_dir, 'Optimizer.dat'))

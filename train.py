#!/usr/bin/python3
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from net import DualPoolBranch, BiBranch, weighted_loss, weights_init
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



def forward_data(net1, net2, net, x1, x2):

    x1 = x1.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
    x2 = x2.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
    x1 = net1(x1)
    x2 = net2(x2)
    y = torch.cat([x1, x2], dim=1)
    y = net(y)
    return y


if __name__ == '__main__':
    args = parse_args()
    logger = WinlaicLogger()
    loss_collector = Averager()
    saver = ModelSaver()
    device = torch.device('cuda:0')
    # train_set, validation_set, test_set = generate_dataset(args.data_dir, args.target_file, part_ratio=(15,5,5))
    train_set, validation_set = generate_SIQAD(args.data_dir, args.target_file)


    ori_net = DualPoolBranch()
    grad_net = DualPoolBranch()
    fusion_net = BiBranch()
    nets = [ori_net, grad_net, fusion_net]
    for net in nets: 
        net.cuda()
        net.apply(weights_init)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_threads,
                                             pin_memory=True, drop_last=True)

    parameters = []
    for net in nets:
        for para in net.parameters():
            parameters.append(para)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    for _i in range(args.epochs):
        for net in nets: net.train()
        for patch, patch_grad, mos in tqdm(train_loader):
            for net in nets: net.zero_grad()
            
            
            x = forward_data(ori_net, grad_net, fusion_net, patch, patch_grad)
            x = x.view(args.batch_size, -1).mean(dim=1)
            y = mos.float().cuda()

            loss = torch.nn.functional.mse_loss(x, y, size_average=True)
            loss.backward()
            loss_collector.add(loss.data)
            optimizer.step()
        
        logger.i=['epoch',_i ,'loss',loss_collector.mean]
        loss_collector.clear()
        if _i != 0 and _i % 10 == 0:
            net.eval()
            with torch.no_grad():

                # Reserve for saving predition vectors.
                y_all = torch.empty(size=(len(validation_set),),dtype=torch.float32)
                y_bar_all = torch.empty(size=(len(validation_set),),dtype=torch.float32)

                # forward on test set
                for i, (eval_img, eval_grad, eval_y) in enumerate(validation_set):
                    y_bar = forward_data(ori_net, grad_net, fusion_net, eval_img, eval_grad).mean()
                    y_all[i] = torch.tensor(eval_y, dtype=torch.float32)
                    y_bar_all[i] = y_bar.cpu()
                
                this_lcc = LCC(y_all, y_bar_all).numpy()
                this_srocc = SROCC(y_all, y_bar_all).numpy()
                logger.w=['LCC', this_lcc]
                logger.w=['SROCC', this_srocc]

                # Save model
                if _i % args.save_freq == 0:
                    logger.w = ['Model saved']
                    save_dir = saver.save_dir(['LCC','%.3f' % float(this_lcc)])
                    torch.save(ori_net.state_dict(), join(save_dir, 'Main.mdl'))
                    torch.save(grad_net.state_dict(), join(save_dir, 'Grad.mdl'))
                    torch.save(fusion_net.state_dict(), join(save_dir, 'Fusion.mdl'))
                    torch.save(optimizer.state_dict(), join(save_dir, 'Optimizer.dat'))

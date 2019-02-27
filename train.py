#!/usr/bin/python3
import argparse
import numpy as np
import torch
import torch.functional as F
from net import deepIQA, weighted_loss, weights_init
import imageio
from utils import *
from dataset import *
from winlaic_utils import WinlaicLogger, Averager, ModelSaver
import pdb
import tqdm
from iqa_utils import LCC, SROCC

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--data-dir',
                        default='/media/cel-door/6030688C30686B4C/winlaic_dataset/IQA/TID2013_new/distorted_images')
    parser.add_argument('-t', '--target-file',
                        default='/media/cel-door/6030688C30686B4C/winlaic_dataset/IQA/TID2013_new/mos_with_names.txt')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5000, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('-j', '--num-threads', default=8)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    logger = WinlaicLogger()
    loss_collector = Averager()
    saver = ModelSaver()
    device = torch.device('cuda:0')
    train_set, validation_set, test_set = generate_dataset(args.data_dir, args.target_file, part_ratio=(15,5,5))
    net = deepIQA()
    net.cuda().train()
    # net.apply(weights_init)
    train_loader = torch.utils.data.DataLoader(train_set, 4, shuffle=True, num_workers=args.num_threads,
                                             pin_memory=True, drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    for _i in range(args.epochs):
        net.train()
        for patch, mos in tqdm.tqdm(train_loader):
            patch = patch.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
            net.zero_grad()
            x, a = net(patch)
            y = mos.float().cuda()
            loss = weighted_loss(x, a, y)
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
                    x_bar, a_bar = net(eval_img)
                    y_bar = x_bar*a_bar
                    y_bar = y_bar.reshape(-1,32).sum(1)/a_bar.reshape(-1,32).sum(1)
                    y_all[i] = torch.tensor(eval_y,dtype=torch.float32)
                    y_bar_all[i] = y_bar.cpu()

                logger.i=['LCC', LCC(y_all, y_bar_all)]
                logger.i=['SROCC', SROCC(y_all, y_bar_all)]
                torch.save(net, join(saver.save_dir(), 'Model.mdl')

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



def forward_data(net1, net2, net, x1, x2):
    x1 = x1.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
    x2 = x2.reshape([-1,32,32,3]).transpose(1,3).transpose(2,3).float().cuda()
    x1 = net1(x1)
    x2 = net2(x2)
    y = torch.cat([x1, x2], dim=1)
    y = net(y)
    return y



if __name__ == '__main__':
    logger = WinlaicLogger()
    loss_collector = Averager()
    saver = ModelSaver()
    device = torch.device('cuda:0')
    # train_set, validation_set, test_set = generate_dataset(args.data_dir, args.target_file, part_ratio=(15,5,5))
    train_set, validation_set = generate_SIQAD(args.data_dir, args.target_file)

    model_save_dir = 'saved_models'

    ori_net = DualPoolBranch()
    grad_net = DualPoolBranch()
    fusion_net = BiBranch()

    latest_model_dir = os.listdir(model_save_dir)
    latest_model_dir.sort(reverse=True)
    latest_model_sub_dir = os.listdir(join(model_save_dir, latest_model_dir[0]))
    latest_model_sub_dir.sort(reverse=True)
    latest_model = join(model_save_dir,latest_model_dir[0],latest_model_sub_dir[0])

    ori_net.load_state_dict(torch.load(join(latest_model, 'Main.mdl')))
    grad_net.load_state_dict(torch.load(join(latest_model, 'Grad.mdl')))
    fusion_net.load_state_dict(torch.load(join(latest_model, 'Fusion.mdl')))

    extract_net = [ori_net, grad_net, fusion_net]

    # Freeze previous network.
    for net in extract_net:
        for para in net.parameters():
            para.requires_grad = False

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
                    torch.save(net, join(save_dir, 'Model.mdl'))
                    torch.save(optimizer, join(save_dir, 'Optimizer.dat'))

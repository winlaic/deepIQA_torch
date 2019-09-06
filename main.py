import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import argparse
from PIL import Image
from numpy.lib.stride_tricks import as_strided
from os.path import join
import tqdm
import json
import pdb
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
from winlaic_utils import removeall, Averager
from tensorboardX import SummaryWriter
import yaml
import os


def extract_patches(img, patch_shape):
    '''Divide image into non-overlapped patches.
    image tensor axes are arranged in form of [H(eight) W(idth) C(hannel)].
    '''
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    cropped_shape = list(img.shape)
    cropped_shape[0] -= cropped_shape[0] % patch_shape[0]
    cropped_shape[1] -= cropped_shape[1] % patch_shape[1]
    # Draw 3D graph of the data, calculate step of jump.
    new_strides = (
        3*img.shape[1]*patch_shape[0], 
        3*patch_shape[1], 
        3*img.shape[1], 
        3, 
        1,
    )
    new_shape = (
        cropped_shape[0] // patch_shape[0],
        cropped_shape[1] // patch_shape[1],
        patch_shape[0],
        patch_shape[1],
        3,
    )
    return as_strided(img, shape=new_shape, strides=new_strides)



class IQADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_list, index_column_img, index_column_target, number_patch, patch_shape, phase_patch='random'):
        self._number_patch = number_patch
        self._dataset_list = dataset_list
        self._patch_indexes_fixed = phase_patch
        if self._patch_indexes_fixed == 'fixed':
            self._fixed_choice_indexes = []
        self._data_dir = data_dir
        self._patch_shape = patch_shape
        self._index_column_img = index_column_img
        self._index_column_target = index_column_target
        self.data_pool = []
        print('Initializing data pool...')
        self.init_data_pool()
        
    def init_data_pool(self):
        for item in tqdm.tqdm(self._dataset_list):
            img = Image.open(join(self._data_dir, item[self._index_column_img]))
            img = extract_patches(img, self._patch_shape)
            self.data_pool.append(img)
            if self._patch_indexes_fixed == 'fixed':
                self._fixed_choice_indexes.append(
                    np.random.choice(np.prod(img.shape[0:2]), self._number_patch, replace=False)
                )
            
    
    def __getitem__(self, index):
        img = self.data_pool[index]
        img = img.reshape(-1, *self._patch_shape, 3)
        number_total_patch = img.shape[0]
        if self._patch_indexes_fixed == 'fixed':
            choice_index = self._fixed_choice_indexes[index]
        else:
            choice_index = np.random.choice(number_total_patch, self._number_patch, replace=False)

        img = img[choice_index, :, :, :]
        target = self._dataset_list[index][self._index_column_target]

        img = torch.tensor(img).transpose(1, 3).transpose(2, 3).float()
        # img = img / 255.0

        target = torch.tensor(target).float()

        return img, target
        
    def __len__(self):
        return len(self._dataset_list)

class DoublePool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x_max = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x_mean = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return torch.cat([x_max, x_mean], 1)

class DoubleConv(nn.Module):
    def __init__(self, N_in, N_out):
        super().__init__()
        self.conv_1 = nn.Conv2d(N_in, N_out, 3, padding=1)
        self.conv_2 = nn.Conv2d(N_out, N_out, 3, padding=1)
        # self.pool = DoublePool()
    def forward(self, x):
        x = self.conv_1(x)
        x = nn.functional.relu(x, inplace=True)
        # x = nn.functional.leaky_relu(x, inplace=True, negative_slope=0.1)
        x = self.conv_2(x)
        x = nn.functional.relu(x, inplace=True)
        # x = nn.functional.leaky_relu(x, inplace=True, negative_slope=0.1)
        # return self.pool(x)
        return F.max_pool2d(x, kernel_size=2, stride=2)

class deeqIQA(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_convs = nn.Sequential(
            DoubleConv(  3,  32),
            DoubleConv( 32,  64),
            DoubleConv( 64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        )
        self.fc_1 = nn.Linear(512, 512)
        self.drop_out = nn.Dropout()
        self.fc_2 = nn.Linear(512,   1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.double_convs(x)
        x = x.view(-1, 512)
        x = self.fc_1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.drop_out(x)
        # x = nn.functional.leaky_relu(x, inplace=True, negative_slope=0.1)
        x = self.fc_2(x)
        return x




class EncodeConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, maxpool=True):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=padding)
        self._maxpool = maxpool
    
    def forward(self, x):
        x = F.relu(self.conv_1(x), inplace=True)
        x = F.relu(self.conv_2(x), inplace=True)
        if self._maxpool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x
        
        
def train(args):

    removeall('runs')
    tb = SummaryWriter()
    for key, value in args.items():
        tb.add_text(key, str(value))
        print('{}: {}'.format(key, value))

    print('Start tensorboard.')
    print('-----------------------------------')
    print('tensorboard --logdir {}'.format(join(os.getcwd(), 'runs')))
    print('-----------------------------------')

    loss_collector = Averager()

    net = deeqIQA()
    net.cuda()

    with open(args['train_metadata_file']) as f:
        train_list = json.load(f)
    with open(args['validate_metadata_file']) as f:
        validate_list = json.load(f)

    
    train_dataset = IQADataset(
        args['data_dir'], 
        train_list, 
        args['index_column_img'], 
        args['index_column_target'], 
        number_patch=args['number_patch'], 
        patch_shape=(
            args['size_patch'], 
            args['size_patch']
        )
    )
    validate_dataset = IQADataset(
        args['data_dir'], 
        validate_list, 
        args['index_column_img'], 
        args['index_column_target'], 
        number_patch=32, 
        patch_shape=(
            args['size_patch'], 
            args['size_patch']
        ),
        phase_patch='fixed'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        args['number_batch'], 
        shuffle=True, 
        num_workers=8,
        pin_memory=True, 
        drop_last=True
    )

    # optimizer = torch.optim.SGD(net.parameters(), lr=args['learning_rate'], weight_decay=args['regulation'])

    optimizer = torch.optim.Adam(net.parameters(), lr=args['learning_rate'], weight_decay=args['regulation'])
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=20, verbose=True, 
        threshold=0.005, threshold_mode='abs', cooldown=10, min_lr=0, eps=1e-08
    )


    print('Start training...')
    for i_epoch in tqdm.trange(args['number_epoch']):
        net.train()
        for patches, moses in train_loader:
            net.zero_grad()
            patches = patches.view(-1, 3, args['size_patch'], args['size_patch']).cuda()
            moses_bar = net(patches)
            moses_bar = moses_bar.squeeze()
            moses_ = moses.repeat(args['number_patch'], 1).transpose(0, 1).flatten().cuda()
            loss = nn.functional.l1_loss(
                moses_bar, 
                moses_
            )
            
            loss_collector.add(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        
        tb.add_scalar('train/loss', loss_collector.mean, i_epoch)
        loss_collector.clear()

        if (i_epoch + 1)%args['validate_period'] == 0:
            net.eval()
            with torch.no_grad():
                # Evaluate on validation set.
                mos_bar_validate = torch.empty(size=(len(validate_dataset),),dtype=torch.float32)
                mos_validate = torch.empty(size=(len(validate_dataset),),dtype=torch.float32)
                for i_patch, (patches, mos) in enumerate(validate_dataset):
                    mos_bar_validate[i_patch] = net(patches.cuda()).mean().cpu()
                    mos_validate[i_patch] = mos
                mos_bar_validate = mos_bar_validate.numpy()
                mos_validate = mos_validate.numpy()
                srocc_validate = spearmanr(mos_bar_validate, mos_validate)[0]
                plcc_validate  = pearsonr(mos_bar_validate, mos_validate)[0]

                # Evaluate on trainning set.
                mos_bar_fit = torch.empty(size=(len(train_dataset),),dtype=torch.float32)
                mos_fit = torch.empty(size=(len(train_dataset),),dtype=torch.float32)
                for i_patch in range(len(train_dataset)):
                    mos_bar_fit[i_patch] = net(train_dataset[i_patch][0].cuda()).mean().cpu()
                    mos_fit[i_patch] = train_dataset[i_patch][1]
                mos_bar_fit = mos_bar_fit.numpy()
                mos_fit = mos_fit.numpy()
                srocc_fit = spearmanr(mos_bar_fit, mos_fit)[0]
                plcc_fit = pearsonr(mos_bar_fit, mos_fit)[0]

                schedular.step(srocc_fit)

                tb.add_scalars('validate/srocc', {'validate_set': srocc_validate, 'train_set': srocc_fit}, i_epoch)
                tb.add_scalars('validate/plcc',  {'validate_set': plcc_validate,  'train_set': plcc_fit},  i_epoch)
                
                for name, param in net.named_parameters():
                    tb.add_histogram(name, param.clone().cpu().data.numpy(), i_epoch)

        

def main():
    with open('params.yml') as f:
        args = yaml.load(f)
    train(args)



if __name__ == '__main__': main()
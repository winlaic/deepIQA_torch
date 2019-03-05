import numpy as np
import torch, torch.utils.data
import torch.functional as F
from utils import *
import imageio
import csv
import random
from os.path import join
import pdb
import os, re

class TID_Dataset(torch.utils.data.Dataset):

    def __init__(self,file_path_list,mos_list, patch_size=32):
        self.file_path_list = file_path_list
        self.mos_list = mos_list
        self.patch_size = patch_size

    def __getitem__(self, index):
        img = extract_patches(imageio.imread(self.file_path_list[index]))
        img = np.reshape(img,[-1,32,32,3])
        img = img[np.random.choice(img.shape[0], self.patch_size, replace=False)]
        mos = self.mos_list[index]
        return torch.tensor(img), mos

    def __len__(self):
        return len(self.mos_list)

class SIQAD(torch.utils.data.Dataset):

    def __init__(self, file_path_list, mos_list, patch_size=32):
        self.file_path_list = file_path_list
        self.mos_list = mos_list
        self.patch_size = patch_size

    def __getitem__(self, index):
        img = extract_patches(imageio.imread(self.file_path_list[index]))
        img = np.reshape(img, [-1,32,32,3])
        img = img[np.random.choice(img.shape[0], self.patch_size, replace=False)]
        mos = self.mos_list[index]
        return torch.tensor(img), mos

    def __len__(self):
        return len(self.file_path_list)

def generate_dataset(image_path, mos_file, part_ratio=(15,5,5)):
    with open(mos_file) as f:
        mos_file = csv.reader(f,delimiter=' ')
        mos_file = list(mos_file)
    mos_file = [[float(item[0]), item[1].lower()] for item in mos_file]
    n_image = len(mos_file)
    n_training = int(np.floor(n_image * (part_ratio[0]/sum(part_ratio))))
    n_validation = int(np.floor(n_image * (part_ratio[1]/sum(part_ratio))))
    n_test = n_image - n_training - n_validation
    random.shuffle(mos_file)
    training_file_paths = [join(image_path,item[1]) for item in mos_file[0:0 + n_training]]
    training_mos = [item[0] for item in mos_file[0:0 + n_training]]
    return TID_Dataset(training_file_paths, training_mos),\
        TID_Dataset([join(image_path,item[1]) for item in mos_file[n_training:n_training + n_validation]], [item[0] for item in mos_file[n_training:n_training + n_validation]]),\
        TID_Dataset([join(image_path,item[1]) for item in mos_file[n_validation:n_validation + n_test]], [item[0] for item in mos_file[n_validation:n_validation + n_test]])
    
def generate_SIQAD(image_path, dmos_file, part_ratio=(8,2)):

    def sort_id(x):
         id_tuple = re.findall('^cim(\d+)_(\d)_(\d).bmp$',x)[0]
         return int(id_tuple[0]+id_tuple[1]+id_tuple[2])
        
    image_file_list = os.listdir(image_path)
    image_file_list.sort(key=sort_id)
    image_file_list = np.array(image_file_list)

    with open(dmos_file) as f:
        dmos_file = csv.reader(f)
        dmos = np.array(list(dmos_file),dtype=np.float32).transpose(1,0).reshape(-1)
    mos = (100.0 - dmos)/10.0 # reverse and rescale from 0~100 to 0~10.

    n_image = len(mos)
    n_training = int(np.floor(n_image * (part_ratio[0]/sum(part_ratio))))

    shuffle_index = np.random.choice(n_image,n_image,replace=False)
    image_file_list = image_file_list[shuffle_index]
    image_file_list = [str(item) for item in image_file_list]
    for i in range(n_image):
        image_file_list[i] = join(image_path, image_file_list[i])
    mos = mos[shuffle_index]
    
    training_image_file = image_file_list[0:n_training]
    training_image_mos = mos[0:n_training]

    testing_image_file = image_file_list[n_training:]
    testing_image_mos = mos[n_training:]

    return SIQAD(training_image_file, training_image_mos), \
            SIQAD(testing_image_file, testing_image_mos)
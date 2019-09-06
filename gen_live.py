import os
from os.path import join
import numpy as np
import json
from scipy.io import loadmat
import csv
import random



train_ratio = 0.8
dataset_dir = '/path/to/LIVE/root'
train_file_name = '/path/to/save/metadata/LIVE/metadata_train.json'
validate_file_name = '/path/to/save/metadata/LIVE/metadata_validate.json'
deprecated_class = []
deprecated_img = []

distoration_classes = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
distoration_image_number = [227, 233, 174, 174, 174]
reference_imgs = [item for item in os.listdir(join(dataset_dir, 'refimgs')) if item.split(sep='.')[-1] == 'bmp']


listt = lambda x: list(map(list, zip(*x)))

if __name__ == '__main__':
    
    dmos_of_distoration_types = []
    is_original_image = []
    labels = loadmat(join(dataset_dir, 'dmos.mat'))

    for i in range(len(distoration_image_number)):
        previous_index = sum(distoration_image_number[0:i])
        next_index = previous_index + distoration_image_number[i]
        dmos_of_distoration_types.append(labels['dmos'].squeeze()[previous_index:next_index].tolist())
        is_original_image.append(labels['orgs'].squeeze()[previous_index:next_index].tolist())
    
    infos = []
    for i, item in enumerate(distoration_classes):
        with open(join(dataset_dir, item, 'info.txt')) as f:
            info = list(csv.reader(f, delimiter=' '))
            info = [iitem for iitem in info if len(iitem) != 0]
            info = listt(listt(info)[0:2])
            info.sort(key=lambda x: int(x[1].split(sep='.')[0][3:]))
            info = listt(info)
            info.append(dmos_of_distoration_types[i])
            info.append(is_original_image[i])
            info.insert(1, list(map(lambda x: join('refimgs', x), info[0])))
            info[2] = list(map(lambda x: join(distoration_classes[i], x), info[2]))
            info.insert(1, [item for _ in range(len(info[0]))])
            info = listt(info)
            infos += info


    infos_distored_only = [item for item in infos if item[-1]==0]
    infos_distored_only.sort(key=lambda x: x[0])

    n_images = len(reference_imgs)
    n_train_images = int(n_images * train_ratio)
    n_validate_images = n_images - n_train_images

    # distored_original_img_names = listt(infos_distored_only)[0]
    # number_of_images = [distored_original_img_names.count(item) for item in reference_imgs]

    random.shuffle(reference_imgs)
    images_for_train = reference_imgs[0:n_train_images]
    images_for_validate = reference_imgs[n_train_images:]

    infos_train = [item for item in infos_distored_only if item[0] in images_for_train]
    infos_validate = [item for item in infos_distored_only if item[0] in images_for_validate]

    with open(train_file_name, 'w') as f:
        json.dump(infos_train, f)
    with open(validate_file_name, 'w') as f:
        json.dump(infos_validate, f)

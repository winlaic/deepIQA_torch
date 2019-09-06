import csv
import os
import numpy as np
import json


train_ratio = 0.8
labels_file_name = '/path/to/TID2013/mos_with_names.txt'
train_file_name = '/path/to/save/metadata/TID2013/metadata_train.json'
validate_file_name = '/path/to/save/metadata/TID2013/metadata_validate.json'
deprecated_class = []
deprecated_img = []

listt = lambda x:[[y[i] for y in x] for i in range(len(x[0]))]

if __name__ == '__main__':
    
    with open(labels_file_name) as f:
        reader = csv.reader(f, delimiter = ' ')
        reader = list(reader)
    item_list = []
    for item in reader:
        index = item[1].split(sep = '.')[0][1:].split(sep = '_')
        index = [int(iitem) for iitem in index]
        item[0] = float(item[0])
        item_list.append(index + item)

    item_list_deprecation = [item for item in item_list 
                                if  (item[0] not in deprecated_img) 
                                and (item[1] not in deprecated_class)]
    
    valid_imgs = list(set(listt(item_list_deprecation)[0]))
    valid_imgs.sort()
    number_imgs = valid_imgs.__len__()
    number_train_imgs = int(number_imgs*train_ratio)
    valid_imgs = np.array(valid_imgs)
    random_index = np.random.choice(number_imgs, number_imgs, replace=False)

    indexs_train_imgs = valid_imgs[random_index[0:number_train_imgs]]
    indexs_validate_imgs = valid_imgs[random_index[number_train_imgs:]]

    item_list_train = [item for item in item_list_deprecation
                        if item[0] in indexs_train_imgs]
    item_list_validate = [item for item in item_list_deprecation
                        if item[0] in indexs_validate_imgs]                    

    with open(train_file_name, 'w') as f:
        json.dump(item_list_train, f)

    with open(validate_file_name, 'w') as f:
        json.dump(item_list_validate, f)
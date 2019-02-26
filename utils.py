#!/usr/bin/python3
import numpy as np
from numpy.lib.stride_tricks import as_strided
import imageio
import numbers
from matplotlib import pyplot as plt
import torch
def extract_patches(arr, patch_shape=(32,32,3), extraction_step=32):
    arr_ndim = arr.ndim

    extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    
    # patches = as_strided(arr, shape=shape)
    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


if __name__ == '__main__':
    img = imageio.imread('/media/winlaic/Backup/图片/3a56fd18972bd407f2d476aa79899e510eb309cc.jpg')
    # img = img[:,:,0:3]
    img = extract_patches(img)
    # plt.imshow(img[11,8,0,:,:,:])
    # plt.figure()
    # plt.imshow(img[12,8,0,:,:,:])
    pass
    # plt.show()
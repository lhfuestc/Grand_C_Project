# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:18:56 2019

@author: 144-Tesla-K20
"""
from __future__ import absolute_import
from skimage import measure, transform
from datetime import datetime
from models import model_2d
from scipy import misc


import tensorflow as tf
import numpy as np
import termcolor
import optimize
import options
import random
import loss
import time
import os


import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

norm_8 = 255.0

norm_16 = 60000.0


def calc_psnr_and_ssim(pred_batch, real_batch):
    """Calculate psnr and ssim for a batch of examples.
    
    Args:
    - real_batch:
        A batch of ground truth examples.
    - pred_batch:
        A batch of predicated examples corresponding to real_batch.
        
    Returns:
       Average psnr and ssim values for this batch.
    """
    mean_psnr = 0.0
    mean_ssim = 0.0
    num_example = pred_batch.shape[0]
    for i in range(num_example):
        real, pred = real_batch[i, :, :, :], pred_batch[i,:,:,:]
        real = real.astype(np.float32)
        pred = pred.astype(np.float32)
        mean_psnr += measure.compare_psnr(real, pred)
        mean_ssim += measure.compare_ssim(real, pred, multichannel = True)
    
    mean_psnr = mean_psnr/num_example
    mean_ssim = mean_ssim/num_example
    
    return mean_psnr, mean_ssim

    
def read_images(seg_dir, lab_dir):
    seg_names = sorted(os.listdir(seg_dir))
    lab_names = sorted(os.listdir(lab_dir))
    if len(seg_names) != len(lab_names):
        raise ValueError("The number of SEG and LAB images does not match!")
    
    seg_images = []
    lab_images = []
    for i in range(len(seg_names)):
        # check whether seg&lab image names match each other
        seg_name = seg_names[i].split('.')[0].split('_')[-1]
        lab_name = lab_names[i].split('.')[0].split('_')[-1]
        if seg_name != lab_name:
            errstr = "seg image name %s does not match lab image name %s!" % (seg_name, lab_name)
            raise ValueError(errstr)
        # read seg & lab image pair
        seg_im = (np.load(seg_dir + seg_names[i]) / norm_16).astype(np.float32)
        seg_im = transform.resize(seg_im, [256, 256], order = 3, mode = 'constant', cval = 0)
        lab_im = (misc.imread(lab_dir + lab_names[i]) / norm_16).astype(np.float32)
        lab_im = transform.resize(lab_im, [256, 256], order = 3, mode = 'constant', cval = 0)
        seg_im = seg_im[:, :, np.newaxis]
        lab_im = lab_im[:, :, np.newaxis]
        seg_images.append(seg_im)
        lab_images.append(lab_im)
    return seg_images, lab_images


def get_valid_batch(seg_images, lab_images):
    num_img = len(seg_images)
    assert num_img == len(lab_images), "valid set size error!"
    seg_batch = np.zeros([num_img, image_H, image_W, 1], dtype = np.float32)
    lab_batch = np.zeros([num_img, label_H, label_W, 1], dtype = np.float32)
    for i in range(num_img):
        seg_image = seg_images[i]
        lab_image = lab_images[i]
        seg_batch[i, :, :, :] = seg_image
        lab_batch[i, :, :, :] = lab_image
    return seg_batch, lab_batch
                                                                                                                                                                                                                                     

def get_train_batch(seg_images, lab_images):
    num_img = len(seg_images)
#    seg_batch = np.zeros([num_img, image_H, image_W, 1], dtype = np.float32)
#    lab_batch = np.zeros([num_img, label_H, label_W, 1], dtype = np.float32)
    seg_batch = np.zeros([batch_size, inpRow, inpCol, 1], dtype = np.float32)
    lab_batch = np.zeros([batch_size, inpRow, inpCol, 1], dtype = np.float32)
    for i in range(num_img):
        # generate an image patch randomly
        seg_image = seg_images[i]
        lab_image = lab_images[i]
        
        gH = np.random.randint(low = 0, high = seg_image.shape[0] - inpRow +1)
        gW = np.random.randint(low = 0, high = seg_image.shape[1] - inpCol +1)
        cH, cW = gH, gW
        seg_patch = seg_image[gH:gH + inpRow, gW:gW + inpCol, :]
        lab_patch = lab_image[cH:cH + inpRow, cW:cW + inpCol, :]
        
        # do some augmentation here Flip + Rotate
        rot = np.random.randint(low = 2, high = 9)
        if rot == 2:    # up-down flip
            seg_patch = seg_patch[::-1, :, :]
            lab_patch = lab_patch[::-1, :, :]
        elif rot == 3:  # left-right flip
            seg_patch = seg_patch[:, ::-1, :]
            lab_patch = lab_patch[:, ::-1, :]
        elif rot == 4:  # up-down + left-right flip
            seg_patch = seg_patch[::-1, :, :]
            lab_patch = lab_patch[::-1, :, :]
        elif rot == 5:  # transpose
            seg_patch = np.transpose(seg_patch, [1, 0, 2])
            lab_patch = np.transpose(lab_patch, [1, 0, 2])
        elif rot == 6:  # transpose + up-down flip
            seg_patch = np.transpose(seg_patch, [1, 0, 2])[::-1, :, :]
            lab_patch = np.transpose(lab_patch, [1, 0, 2])[::-1, :, :]
        elif rot == 7:  # transpose + left-right flip
            seg_patch = np.transpose(seg_patch, [1, 0, 2])[:, ::-1, :]
            lab_patch = np.transpose(lab_patch, [1, 0, 2])[:, ::-1, :]
        else:           # transpose + up-down + left-right flip
            seg_patch = np.transpose(seg_patch, [1, 0, 2])[::-1, ::-1, :]
            lab_patch = np.transpose(lab_patch, [1, 0, 2])[::-1, ::-1, :]
            
        seg_batch[i, :, :, :] = seg_patch
        lab_batch[i, :, :, :] = lab_patch
    
    return seg_batch, lab_batch
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:55:31 2019

@author: 64360
"""
'''
用于随机划分 训练集，验证集，测试集。
'''
import os
import numpy as np
src_dir_DM = 'datasets/CT/DM/train'
dst_dir_DM = 'datasets/CT/DM/test'

src_dir_GT = 'datasets/CT/GT/train'
dst_dir_GT = 'datasets/CT/GT/test'
for i in range(100):
    image_name_list = os.listdir(src_dir_DM)
    label_name_list = os.listdir(src_dir_GT)
    indx = np.random.randint(len(image_name_list))
    src_img_path = os.path.join(src_dir_DM, image_name_list[indx])
    src_lab_path = os.path.join(src_dir_GT, label_name_list[indx])
    
    dst_img_path = os.path.join(dst_dir_DM, image_name_list[indx])
    dst_lab_path = os.path.join(dst_dir_GT, label_name_list[indx])
    
    os.rename(src_img_path, dst_img_path)
    os.rename(src_lab_path, dst_lab_path)
    print(i)
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 14:46:12 2019

@author: 64360
"""
'''
将dicom 转为 npy
'''
import numpy as np
import pydicom
import os
data_dir = 'CT_data_batch1/1/DICOM_anon'
save_dir = 'datasets/CT/DM/train'
patient_id = data_dir.split('/')[1]
img_name_list = os.listdir(data_dir)
for i in range(len(img_name_list)):
    data_path = os.path.join(data_dir, img_name_list[i])
    ds = pydicom.read_file(data_path)
    image_array = ds.pixel_array
    save_path = os.path.join(save_dir, 'CT' + '_' + 'DM' + '_' + patient_id + '_' + str(i) + '.npy')
    np.save(save_path, image_array)
    print(i)


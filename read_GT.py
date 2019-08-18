# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:01:03 2019

@author: 64360
"""
from scipy import misc
import os
data_dir = 'CT_data_batch1/19/Ground'
save_dir = 'datasets/CT/GT/train'
patient_id = data_dir.split('/')[1]
img_name_list = os.listdir(data_dir)
for i in range(len(img_name_list)):
    data_path = os.path.join(data_dir, img_name_list[i])
    image = misc.imread(data_path)
    save_path = os.path.join(save_dir, 'CT' + '_' + 'GT' + '_' + patient_id + '_' + str(i) + '.png')
    misc.imsave(save_path, image)
    print(i)

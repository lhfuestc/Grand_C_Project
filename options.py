# -*- coding: utf-8 -*-
"""
Created on Sat Des 3 19:56:38 2018
@author: LHF
"""
from __future__ import absolute_import
from __future__ import division
import argparse

parser = argparse.ArgumentParser()
#############################################################################################################
#                                          Global Constants                                                 #
#############################################################################################################
DEFAULT_TRAIN_SEG_DIR = 'datasets/CT/DM/train_pps/'
DEFAULT_TRAIN_LAB_DIR = 'datasets/CT/GT/train/'
DEFAULT_VALID_SEG_DIR = 'datasets/CT/DM/valid_pps/'
DEFAULT_VALID_LAB_DIR = 'datasets/CT/GT/valid/'
DEFAULT_TEST_SEG_DIR = 'datasets/CT/DM/test_pps/'
DEFAULT_TEST_LAB_DIR = 'datasets/CT/GT/test/'

img_H, img_W = 256, 256

lab_H, lab_W = 256, 256

InpRow = 256

InpCol = 256

CHANNELS = 1

WEIGHT_DECAY = 0.0

TOTAL_LOSS_COLLECTION = 'my_total_losses'


##########################################################################################################
#                                               Data                                                     #
##########################################################################################################
parser.add_argument('--train_seg_dir', type = str, default = DEFAULT_TRAIN_SEG_DIR,
                    help = 'Path to LR images for training.')
parser.add_argument('--train_lab_dir', type = str, default = DEFAULT_TRAIN_LAB_DIR,
                    help = 'Path to HR images for training.')
parser.add_argument('--valid_seg_dir', type = str, default = DEFAULT_VALID_SEG_DIR,
                    help = 'Path to LR images for validation.')
parser.add_argument('--valid_lab_dir', type = str, default = DEFAULT_VALID_LAB_DIR,
                    help = 'Path to HR images for validation.')
parser.add_argument('--test_seg_dir', type = str, default = DEFAULT_TEST_SEG_DIR,
                    help = 'Path to LR images for testing.')
parser.add_argument('--test_lab_dir', type = str, default = DEFAULT_TEST_LAB_DIR,
                    help = 'Path to HR images for testing.')

##########################################################################################################
#                                               Train                                                    #
##########################################################################################################
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'Number of examples to process in a batch.')
parser.add_argument('--use_fp16', type = bool, default = False,
                    help = 'Train model using float16 data type.')
parser.add_argument('--xavier_init', type = bool, default = True,
                    help = 'whether to initialize params with Xavier method.')
parser.add_argument('--max_steps', type = int, default = 201000,
                    help = 'Maximum of the number of steps to train.')
parser.add_argument('--learning_rate', type = float, default = 0.00005,
                    help = 'The learning rate for optimizer.')
parser.add_argument('--train_log_freq', type = int, default = 100,
                    help = 'How often to log results to the console when training.')
parser.add_argument('--decay_steps', type = int, default = 100000,
                    help = 'How many steps to decay learning rate.')
parser.add_argument('--decay_rate', type = float, default = 0.9,
                    help = 'Decay rate for learning rate.')
parser.add_argument('--train_log_dir', type = str, default = 'train_logs/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--use_mridata_train', type = bool, default = True,
                    help = 'Train model using mri data.')
parser.add_argument('--normalization', type = bool, default = True,
                    help = 'Whether to normalize training examples.')
parser.add_argument('--sub_mean', type = bool, default = True ,
                    help = 'Whether to subtract the average RGB values of the whole dataset.')
parser.add_argument('--train_from_exist', type = bool, default = False,
                    help = 'Whether to train model from pretrianed ones.')
parser.add_argument('--exist_model_dir', type = str, default = 'train_logs/',
                    help = 'Directory where to load pretrianed models.')

##########################################################################################################
#                                                 Valid                                                  #
##########################################################################################################
parser.add_argument('--valid_log_dir', type = str, default = 'records/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--valid_image_save_dir', type = str, default = 'valid_saved_images/',
                    help = 'Directory where to save predicted HR patches.')
parser.add_argument('--valid_save_image', type = bool, default = True,
                    help = 'Whether to save the result images when do evalution.')

##########################################################################################################
#                                                 Test                                                  #
##########################################################################################################
parser.add_argument('--test_log_dir', type = str, default = 'records/',
                    help = 'Directory where to write event logs and checkpoints.')
parser.add_argument('--test_image_save_dir', type = str, default = 'test_saved_images/',
                    help = 'Directory where to save predicted HR patches.')
parser.add_argument('--test_image_save_dir_label', type = str, default = 'test_saved_images/label_dir/',
                    help = 'Directory where to save predicted HR patches.')
parser.add_argument('--test_image_save_dir_pred', type = str, default = 'test_saved_images/pred_dir/',
                    help = 'Directory where to save predicted HR patches.')
parser.add_argument('--test_model_dir', type = str, default = 'train_logs/',
                    help = 'Directory where the model that needs evaluation is saved.')
parser.add_argument('--test_save_image', type = bool, default = True,
                    help = 'Whether to save the result images when do testing.')

#%%
params = parser.parse_args()





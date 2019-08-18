# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:10:01 2018
@author: zxlation
@author: zyl
"""

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import argparse
#import cv2
parser = argparse.ArgumentParser()

# Basic Model Arguments/Parameters
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'Number of examples to process in a batch.')
parser.add_argument('--use_fp16', type = bool, default = False,
                    help = 'Train model using float16 data type.')

PARAMS = parser.parse_args()

WEIGHT_DECAY_LAMBDA = 0.000
WEIGHT_DECAY_COLLECTION = "my_losses"

def total_variation_loss(inputs, mode = 'l1'):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2 or
    V(y) = || y_{n+1} - y_n ||_1
    param:
        inputs:
        mode:
    return:
        L1/L2 TV loss
    """
    dy = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dx = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    if mode is 'l2':
        size_dy = tf.size(dy, out_type=tf.float32)
        size_dx = tf.size(dx, out_type=tf.float32)
        Loss = tf.nn.l2_loss(dy) / size_dy + tf.nn.l2_loss(dx) / size_dx
    else:
        Loss = tf.reduce_mean(tf.abs(dy)) + tf.reduce_mean(tf.abs(dx))
    return Loss

def VariableSummary(var, name):
    """Add a lot of summaries to a tensor."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
def XavierInitializer(prev_units, curr_units, kernel_size, stddev_factor = 1.0):
    """Initialization for CONV2D in the style of Xavier Glorot et al.(2010).
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs.
    ArgS:
        prev_units: The number of channels in the previous layer.
        curr_units: The number of channels in the current layer.
        stddev_factor: 
    Returns:
        Initial value of the weights of the current conv/transpose conv layer.
    """
    stddev = np.sqrt(stddev_factor/(np.sqrt(prev_units*curr_units)*kernel_size*kernel_size))
    
    return tf.truncated_normal_initializer(mean = 0.0, stddev = stddev)


def VariableOnCPU(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if PARAMS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
    return var


def VariableWithWeightDecay(name, shape, wd, is_conv):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        init_mode: Initialization mode for variables: 'conv' and 'convT'

    Returns:
        Variable Tensor
    """
    if is_conv == True:
        initializer = XavierInitializer(shape[2], shape[3], shape[0], stddev_factor = 1.0)
    else:
        initializer = XavierInitializer(shape[3], shape[2], shape[0], stddev_factor = 1.0)
    
    var = VariableOnCPU(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        tf.add_to_collection(WEIGHT_DECAY_COLLECTION, weight_decay)
    return var


def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME', name = name)

def conv_layer(inputs, kernel_shape, wd, name):
    with tf.variable_scope(name):
        W = VariableWithWeightDecay('weights', shape = kernel_shape, wd = wd, is_conv = True)
        b = VariableOnCPU('biases', [kernel_shape[3]], tf.constant_initializer())
        conv = tf.nn.relu(conv2d(inputs, W, name = 'conv_op') + b, name = 'relu_op')
    
    return conv

def norm_layer(inputs, name):
    with tf.variable_scope(name):
        norm = tf.nn.lrn(inputs, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name = name)
        VariableSummary(norm)
    
    return norm

def conv2d_T(x, W, output_shape, name):
    return tf.nn.conv2d_transpose(x, W, output_shape = output_shape, strides = [1, 2, 2, 1], padding = 'SAME', name = name)

def convT_layer(inputs, kernel_shape, output_shape, wd, name):
    with tf.variable_scope(name):
        W = VariableWithWeightDecay('weights_T', shape = kernel_shape, wd = wd, is_conv = False)
        b = VariableOnCPU('biases_T', [kernel_shape[2]], tf.constant_initializer())
        convT = tf.nn.relu(conv2d_T(inputs, W, output_shape = output_shape, name = 'conv_T') + b, name = 'relu_op')
        
    return convT

def max_pooling_2x2(x, name):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def UNet(inputs):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''
    
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    
    # inputs: [BATCH_SIZE, 256, 256, 1]
    
    in_maps = int(inputs.get_shape()[3])
    kernel_size = 3
    
    inputs = conv_layer(inputs, [kernel_size, kernel_size, in_maps, 64],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer1')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer2')
    conv2 = inputs
    layer2_shape = tf.shape(inputs)
    
    
    # pool3: [BATCH_SIZE, 256, 256, 64]
    #        [BATCH_SZIE, 128, 128, 128]
    inputs = max_pooling_2x2(inputs, 'pool_layer3')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 128],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer4')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer5')
    conv5 = inputs
    layer5_shape = tf.shape(inputs)
    
    
    # pool6: [BATCH_SIZE, 128, 128, 128]
    #        [BATCH_SZIE, 64, 64, 256]
    inputs = max_pooling_2x2(inputs, 'pool_layer6')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 256],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer7')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256],
                       WEIGHT_DECAY_LAMBDA, 'conv_layer8')
    conv8 = inputs
    layer8_shape = tf.shape(inputs)
    
    
    # pool9: [BATCH_SIZE, 64, 64, 256]
    #        [BATCH_SZIE, 32, 32, 512]
    inputs = max_pooling_2x2(inputs, 'pool_layer9')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer10')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer11')
    conv11 = inputs
    layer11_shape = tf.shape(inputs)
    
    
    # pool12: [BATCH_SIZE, 32, 32, 512]
    #         [BATCH_SZIE, 16, 16, 1024]
    inputs = max_pooling_2x2(inputs, 'pool_layer12')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 1024],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer13')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 1024, 1024],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer14')
    
    
    # convT15: [BATCH_SIZE, 16, 16, 1024]
    #          [BATCH_SZIE, 32, 32, 512]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 512, 1024], layer11_shape,
                         WEIGHT_DECAY_LAMBDA, 'convT_layer15')
    inputs = conv_layer(tf.concat([inputs, conv11], 3),[kernel_size, kernel_size, 1024, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer16')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer17')
    
    
    # convT18: [BATCH_SIZE, 32, 32, 512]
    #          [BATCH_SZIE, 64, 64, 256]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 256, 512], layer8_shape,
                         WEIGHT_DECAY_LAMBDA, 'convT_layer18')
    inputs = conv_layer(tf.concat([inputs, conv8], 3), [kernel_size, kernel_size, 512, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer19')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer20')
    
    
    # convT21: [BATCH_SIZE, 64, 64, 256]
    #          [BATCH_SZIE, 128, 128, 128]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 128, 256], layer5_shape,
                          WEIGHT_DECAY_LAMBDA, 'convT_layer21')
    inputs = conv_layer(tf.concat([inputs, conv5], 3), [kernel_size, kernel_size, 256, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer22')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer23')
    
    
    # convT24: [BATCH_SIZE, 128, 128, 128]
    #          [BATCH_SZIE, 256, 256, 64]
    inputs = convT_layer(inputs, [kernel_size, kernel_size, 64, 128], layer2_shape,
                          WEIGHT_DECAY_LAMBDA,'convT_layer24')
    inputs = conv_layer(tf.concat([inputs, conv2], 3), [kernel_size, kernel_size, 128, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer25')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer26')
    
    inputs = conv_layer(inputs, [1, 1, 64, 1], WEIGHT_DECAY_LAMBDA, 'conv_layer27')
    
    # Filter exception value
    inputs = tf.minimum(tf.maximum(inputs, 0.0), 1.0)
    
    return inputs


def UNet_without_convT(inputs):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''
    
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    in_maps = int(inputs.get_shape()[3])
    kernel_size = 3
    
    
    # inputs: [BATCH_SIZE, 256, 256, 1]
   
    inputs = conv_layer(inputs, [kernel_size, kernel_size, in_maps, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer1')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer2')
    conv2 = inputs
    
    
    # pool3: [BATCH_SIZE, 256, 256, 64]
    #        [BATCH_SZIE, 128, 128, 128]
    inputs = max_pooling_2x2(inputs, 'pool_layer3')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer4')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer5')
    conv5 = inputs
    
    
    # pool6: [BATCH_SIZE, 128, 128, 128]
    #        [BATCH_SZIE, 64, 64, 256]
    inputs = max_pooling_2x2(inputs, 'pool_layer6')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer7')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer8')
    conv8 = inputs
    
    
    # pool9: [BATCH_SIZE, 64, 64, 256]
    #        [BATCH_SZIE, 32, 32, 512]
    inputs = max_pooling_2x2(inputs, 'pool_layer9')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer10')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer11')
    conv11 = inputs
    
    # pool12: [BATCH_SIZE, 32, 32, 512]
    #         [BATCH_SZIE, 16, 16, 1024]
    inputs = max_pooling_2x2(inputs, 'pool_layer12')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 1024],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer13')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 1024, 1024],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer14')
    
    
    # convT15: [BATCH_SIZE, 16, 16, 1024]
    #          [BATCH_SZIE,  32,  32, 512]
    conv11_shape = tf.shape(conv11)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv11_shape[1], conv11_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 1024, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer15')
    inputs = conv_layer(tf.concat([inputs, conv11], 3), [kernel_size, kernel_size, 1024, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer16')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer17')
    
    
    # convT18: [BATCH_SIZE, 32, 32, 512]
    #          [BATCH_SZIE, 64, 64, 256]
    conv8_shape = tf.shape(conv8)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv8_shape[1], conv8_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer18')
    inputs = conv_layer(tf.concat([inputs, conv8], 3), [kernel_size, kernel_size, 512, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer19')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer20')
    
    
    # convT21: [BATCH_SIZE, 64, 64, 256]
    #          [BATCH_SZIE, 128, 128, 128]
    conv5_shape = tf.shape(conv5)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv5_shape[1], conv5_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer21')
    inputs = conv_layer(tf.concat([inputs, conv5], 3), [kernel_size, kernel_size, 256, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer22')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer23')
    
    
    # convT24: [BATCH_SIZE, 128, 128, 128]
    #          [BATCH_SZIE, 256, 256, 64]
    conv2_shape = tf.shape(conv2)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv2_shape[1], conv2_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer24')
    inputs = conv_layer(tf.concat([inputs, conv2], 3), [kernel_size, kernel_size, 128, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer25')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64],
                        WEIGHT_DECAY_LAMBDA, 'conv_layer26')
    
    # pred_HR: [BATCH_SIZE, 48, 48, 3]
    inputs = conv_layer(inputs, [1, 1, 64, 1], WEIGHT_DECAY_LAMBDA, 'conv_layer27')
    
    # Filter exception value
    inputs = tf.minimum(tf.maximum(inputs, 0.0), 1.0)
    
    return inputs


def train_UNet_depth6(inputs):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''
    
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    
    # lr_images: [BATCH_SIZE, 96, 96, 3]
    conv1 = conv_layer(inputs, [3, 3, 3, 64], WEIGHT_DECAY_LAMBDA, 'conv_layer1')
    conv2 = conv_layer(conv1, [3, 3, 64, 64], WEIGHT_DECAY_LAMBDA, 'conv_layer2')
    
    # pool3: [BATCH_SIZE, 48, 48, 64]
    pool3 = max_pooling_2x2(conv2, 'pool_layer3')
    conv4 = conv_layer(pool3, [3, 3, 64, 128], WEIGHT_DECAY_LAMBDA, 'conv_layer4')
    conv5 = conv_layer(conv4, [3, 3, 128, 128], WEIGHT_DECAY_LAMBDA, 'conv_layer5')
    
    # pool6: [BATCH_SIZE, 24, 24, 128]
    pool6 = max_pooling_2x2(conv5, 'pool_layer6')
    conv7 = conv_layer(pool6, [3, 3, 128, 256], WEIGHT_DECAY_LAMBDA, 'conv_layer7')
    conv8 = conv_layer(conv7, [3, 3, 256, 256], WEIGHT_DECAY_LAMBDA, 'conv_layer8')
    
    # pool9: [BATCH_SIZE, 12, 12, 256]
    pool9 = max_pooling_2x2(conv8, 'pool_layer9')
    conv10 = conv_layer(pool9, [3, 3, 256, 512], WEIGHT_DECAY_LAMBDA, 'conv_layer10')
    conv11 = conv_layer(conv10, [3, 3, 512, 512], WEIGHT_DECAY_LAMBDA, 'conv_layer11')
    
    # pool12: [BATCH_SIZE, 6, 6, 512]
    pool12 = max_pooling_2x2(conv11, 'pool_layer12')
    conv13 = conv_layer(pool12, [3, 3, 512, 1024], WEIGHT_DECAY_LAMBDA, 'conv_layer13')
    conv14 = conv_layer(conv13, [3, 3, 1024, 1024], WEIGHT_DECAY_LAMBDA, 'conv_layer14')
    
    # pool15: [BATCH_SIZE, 3, 3, 1024]
    pool15 = max_pooling_2x2(conv14, 'pool_layer15')
    conv16 = conv_layer(pool15, [3, 3, 1024, 2048], WEIGHT_DECAY_LAMBDA, 'conv_layer16')
    conv17 = conv_layer(conv16, [3, 3, 2048, 2048], WEIGHT_DECAY_LAMBDA, 'conv_layer17')
    
    # convT18: [BATCH_SIZE, 6, 6, 1024]
    convT18 = convT_layer(conv17, [3, 3, 1024, 2048], tf.shape(conv14), WEIGHT_DECAY_LAMBDA, 'convT_layer18')
    conv19 = conv_layer(tf.concat([convT18, conv14], 3), [3, 3, 2048, 1024], WEIGHT_DECAY_LAMBDA, 'conv_layer19')
    conv20 = conv_layer(conv19, [3, 3, 1024, 1024], WEIGHT_DECAY_LAMBDA, 'conv_layer20')
    
    # convT21: [BATCH_SIZE, 12, 12, 512]
    convT21 = convT_layer(conv20, [3, 3, 512, 1024], tf.shape(conv11), WEIGHT_DECAY_LAMBDA, 'convT_layer21')
    conv22 = conv_layer(tf.concat([convT21, conv11], 3), [3, 3, 1024, 512], WEIGHT_DECAY_LAMBDA, 'conv_layer22')
    conv23 = conv_layer(conv22, [3, 3, 512, 512], WEIGHT_DECAY_LAMBDA, 'conv_layer23')
    
    # convT24: [BATCH_SIZE, 24, 24, 256]
    convT24 = convT_layer(conv23, [3, 3, 256, 512], tf.shape(conv8), WEIGHT_DECAY_LAMBDA, 'convT_layer24')
    conv25 = conv_layer(tf.concat([convT24, conv8], 3), [3, 3, 512, 256], WEIGHT_DECAY_LAMBDA, 'conv_layer25')
    conv26 = conv_layer(conv25, [3, 3, 256, 256], WEIGHT_DECAY_LAMBDA, 'conv_layer26')
    
    # convT24: [BATCH_SIZE, 48, 48, 128]
    convT27 = convT_layer(conv26, [3, 3, 128, 256], tf.shape(conv5), WEIGHT_DECAY_LAMBDA,'convT_layer27')
    conv28 = conv_layer(tf.concat([convT27, conv5], 3), [3, 3, 256, 128], WEIGHT_DECAY_LAMBDA, 'conv_layer28')
    conv29 = conv_layer(conv28, [3, 3, 128, 128], WEIGHT_DECAY_LAMBDA, 'conv_layer29')
    
    # convT24: [BATCH_SIZE, 96, 96, 64]
    convT30 = convT_layer(conv29, [3, 3, 64, 128], tf.shape(conv2), WEIGHT_DECAY_LAMBDA,'convT_layer30')
    conv31 = conv_layer(tf.concat([convT30, conv2], 3), [3, 3, 128, 64], WEIGHT_DECAY_LAMBDA, 'conv_layer31')
    conv32 = conv_layer(conv31, [3, 3, 64, 64], WEIGHT_DECAY_LAMBDA, 'conv_layer32')
    
    # pred_HR: [BATCH_SIZE, 96, 96, 3]
    pred_HR = conv_layer(conv32, [1, 1, 64, 3], WEIGHT_DECAY_LAMBDA, 'conv_layer33')
    
    # Filter exception value
    pred_HR = tf.minimum(tf.maximum(pred_HR, 0.0), 1.0)
    
    return pred_HR


def UNet_block(inputs, scope):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''
    
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    in_maps = int(inputs.get_shape()[3])
    kernel_size = 3
    
    conv1 = conv_layer(inputs, [kernel_size, kernel_size, in_maps, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer1')
    conv2 = conv_layer(conv1, [kernel_size, kernel_size, 64, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer2')
    
    pool3 = max_pooling_2x2(conv2, 'pool_layer3')
    conv4 = conv_layer(pool3, [kernel_size, kernel_size, 64, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer4')
    conv5 = conv_layer(conv4, [kernel_size, kernel_size, 128, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer5')
    
    pool6 = max_pooling_2x2(conv5, 'pool_layer6')
    conv7 = conv_layer(pool6, [kernel_size, kernel_size, 128, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer7')
    conv8 = conv_layer(conv7, [kernel_size, kernel_size, 256, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer8')
    
    pool9 = max_pooling_2x2(conv8, 'pool_layer9')
    conv10 = conv_layer(pool9, [kernel_size, kernel_size, 256, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer10')
    conv11 = conv_layer(conv10, [kernel_size, kernel_size, 512, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer11')
    
    pool12 = max_pooling_2x2(conv11, 'pool_layer12')
    conv13 = conv_layer(pool12, [kernel_size, kernel_size, 512, 1024], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer13')
    conv14 = conv_layer(conv13, [kernel_size, kernel_size, 1024, 1024], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer14')
    
    convT15 = convT_layer(conv14, [kernel_size, kernel_size, 512, 1024], tf.shape(conv11), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer15')
    conv16 = conv_layer(tf.concat([convT15, conv11], 3), [kernel_size, kernel_size, 1024, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer16')
    conv17 = conv_layer(conv16, [kernel_size, kernel_size, 512, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer17')
    
    convT18 = convT_layer(conv17, [kernel_size, kernel_size, 256, 512], tf.shape(conv8), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer18')
    conv19 = conv_layer(tf.concat([convT18, conv8], 3), [kernel_size, kernel_size, 512, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer19')
    conv20 = conv_layer(conv19, [kernel_size, kernel_size, 256, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer20')
    
    convT21 = convT_layer(conv20, [kernel_size, kernel_size, 128, 256], tf.shape(conv5), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer21')
    conv22 = conv_layer(tf.concat([convT21, conv5], 3), [kernel_size, kernel_size, 256, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer22')
    conv23 = conv_layer(conv22, [kernel_size, kernel_size, 128, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer23')
    
    convT24 = convT_layer(conv23, [kernel_size, kernel_size, 64, 128], tf.shape(conv2), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer24')
    conv25 = conv_layer(tf.concat([convT24, conv2], 3), [kernel_size, kernel_size, 128, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer25')
    conv26 = conv_layer(conv25, [kernel_size, kernel_size, 64, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer26')
    
    return conv26


def UNet_light_block(inputs, scope):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''
    
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    in_maps = int(inputs.get_shape()[3])
    kernel_size = 3
    
    inputs = conv_layer(inputs, [kernel_size, kernel_size, in_maps, 32], WEIGHT_DECAY_LAMBDA, scope + 'init_conv')
    init_conv = inputs
    
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 32, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer1')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 32, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer2')
    conv2 = inputs
    
    inputs = max_pooling_2x2(inputs, 'pool_layer3')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 32, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer4')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer5')
    conv5 = inputs
    
    inputs = max_pooling_2x2(inputs, 'pool_layer6')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer7')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer8')
    conv8 = inputs
    
    inputs = max_pooling_2x2(inputs, 'pool_layer9')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer10')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer11')
    conv11 = inputs
    
    inputs = max_pooling_2x2(inputs, 'pool_layer12')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer13')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer14')
    
    conv11_shape = tf.shape(conv11)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv11_shape[1], conv11_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 512, 256], WEIGHT_DECAY_LAMBDA, scope + 'convT_layer15')
    inputs = conv_layer(tf.concat([inputs, conv11], 3), [kernel_size, kernel_size, 512, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer16')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer17')
    
    conv8_shape = tf.shape(conv8)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv8_shape[1], conv8_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 256, 128], WEIGHT_DECAY_LAMBDA, scope + 'convT_layer18')
    inputs = conv_layer(tf.concat([inputs, conv8], 3), [kernel_size, kernel_size, 256, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer19')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer20')
    
    conv5_shape = tf.shape(conv5)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv5_shape[1], conv5_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 128, 64], WEIGHT_DECAY_LAMBDA, scope + 'convT_layer21')
    inputs = conv_layer(tf.concat([inputs, conv5], 3), [kernel_size, kernel_size, 128, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer22')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer23')
    
    conv2_shape = tf.shape(conv2)
    inputs = tf.image.resize_nearest_neighbor(inputs, [conv2_shape[1], conv2_shape[2]])
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 64, 32], WEIGHT_DECAY_LAMBDA, scope + 'convT_layer24')
    inputs = conv_layer(tf.concat([inputs, conv2], 3), [kernel_size, kernel_size, 64, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer25')
    inputs = conv_layer(inputs, [kernel_size, kernel_size, 32, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer26')
    
    return inputs + init_conv


def U_block(inputs, scope):
    ''' build the model
    Args:
        GIBBS_images : GIBBS images with size of [batch_size, GIBBS_W, GIBBS_H, Channels]
    Returns:
        predicted_CLEAR: a tensor with size [batch_size, CLEAR_W, CLEAR_H, Channels]
    '''
    
    assert len(inputs.shape) == 4, "The dimension of inputs should be 4!"
    in_maps = int(inputs.get_shape()[3])
    kernel_size = 3
    
    conv1 = conv_layer(inputs, [kernel_size, kernel_size, in_maps, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer1')
    conv2 = conv_layer(conv1, [kernel_size, kernel_size, 32, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer2')
    
    pool3 = max_pooling_2x2(conv2, 'pool_layer3')
    conv4 = conv_layer(pool3, [kernel_size, kernel_size, 32, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer4')
    conv5 = conv_layer(conv4, [kernel_size, kernel_size, 64, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer5')
    
    pool6 = max_pooling_2x2(conv5, 'pool_layer6')
    conv7 = conv_layer(pool6, [kernel_size, kernel_size, 64, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer7')
    conv8 = conv_layer(conv7, [kernel_size, kernel_size, 128, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer8')
    
    pool9 = max_pooling_2x2(conv8, 'pool_layer9')
    conv10 = conv_layer(pool9, [kernel_size, kernel_size, 128, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer10')
    conv11 = conv_layer(conv10, [kernel_size, kernel_size, 256, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer11')
    
    pool12 = max_pooling_2x2(conv11, 'pool_layer12')
    conv13 = conv_layer(pool12, [kernel_size, kernel_size, 256, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer13')
    conv14 = conv_layer(conv13, [kernel_size, kernel_size, 512, 512], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer14')
    
    convT15 = convT_layer(conv14, [kernel_size, kernel_size, 256, 512], tf.shape(conv11), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer15')
    conv16 = conv_layer(tf.concat([convT15, conv11], 3), [kernel_size, kernel_size, 512, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer16')
    conv17 = conv_layer(conv16, [kernel_size, kernel_size, 256, 256], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer17')
    
    convT18 = convT_layer(conv17, [kernel_size, kernel_size, 128, 256], tf.shape(conv8), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer18')
    conv19 = conv_layer(tf.concat([convT18, conv8], 3), [kernel_size, kernel_size, 256, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer19')
    conv20 = conv_layer(conv19, [kernel_size, kernel_size, 128, 128], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer20')
    
    convT21 = convT_layer(conv20, [kernel_size, kernel_size, 64, 128], tf.shape(conv5), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer21')
    conv22 = conv_layer(tf.concat([convT21, conv5], 3), [kernel_size, kernel_size, 128, 64], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer22')
    conv23 = conv_layer(conv22, [kernel_size, kernel_size, 32, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer23')
    
    convT24 = convT_layer(conv23, [kernel_size, kernel_size, 32, 64], tf.shape(conv2), WEIGHT_DECAY_LAMBDA, scope + 'convT_layer24')
    conv25 = conv_layer(tf.concat([convT24, conv2], 3), [kernel_size, kernel_size, 64, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer25')
    conv26 = conv_layer(conv25, [kernel_size, kernel_size, 32, 32], WEIGHT_DECAY_LAMBDA, scope + 'conv_layer26')
    
    return conv26

def train_multi_UNet(inputs, NumBlocks):
    if NumBlocks <= 0:
        print("The number of U-Block should be larger than 0!")
        return
    
    # init mapping
    init_conv = conv_layer(inputs, [3, 3, 1, 32], WEIGHT_DECAY_LAMBDA, 'init_conv')

    for i in range(NumBlocks):
        inputs = tf.concat([inputs, init_conv], 3)
        inputs = UNet_light_block(inputs, "U_block%d" % i)
    
    out_maps = int(inputs.get_shape()[3])
    pred_HR = conv_layer(inputs, [1, 1, out_maps, 1], WEIGHT_DECAY_LAMBDA, '1x1_conv_layer')
    
    pred_HR = tf.minimum(tf.maximum(pred_HR, 0.0), 1.0)
    
    return pred_HR
    

def loss(CLEAR, pred_CLEAR, lamda = 0.001, TV = False):
    """Calculates the loss from the real HR images and the predicted CLEAR images.

    Args:
        CLEAR: the corresponding true CLEAR images, i.e. reference CLEAR images => [b, h, w, c]
        pred_CLEAR: predicted CLEAR images by the model.                     => [b, h, w, c]

    Returns:
        loss: MSE between real CLEAR images and predicted CLEAR images.
    """
    with tf.name_scope('loss'):
        mse_loss = tf.losses.mean_squared_error(CLEAR, pred_CLEAR)
        tf.add_to_collection(WEIGHT_DECAY_COLLECTION, mse_loss)
        tf.summary.scalar(mse_loss.op.name, mse_loss)
        if (TV):
            tv = total_variation_loss(pred_CLEAR)       
            tv_loss = tf.reduce_sum(tv)*lamda
            tf.add_to_collection(WEIGHT_DECAY_COLLECTION, tv_loss)
            tf.summary.scalar(tv_loss.op.name, tv_loss)
            
        total_losses = tf.add_n(tf.get_collection(WEIGHT_DECAY_COLLECTION))
            
    return total_losses


def optimize(loss, learning_rate, global_step):
    """Sets up the training Ops.
    
    Args:
        loss: Loss tensor, from loss().
        lr: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    decayed_lr = tf.train.exponential_decay(learning_rate,
                                            global_step,
                                            decay_steps = 500,
                                            decay_rate = 0.9,
                                            staircase = True)
    with tf.name_scope('learning_rate'):
        tf.summary.scalar('learning_rate', decayed_lr)
        tf.summary.histogram('histogram', decayed_lr)
        
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate = decayed_lr)
        
        # with a global_step to track the global step.
        train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op


def evaluation(CLEAR, pred_CLEAR):
    """Evaluate the quality of the predicted CLEAR images at predicting the CLEAR iamges.

    Args:
        CLEAR: The real CLEAR images.
        pred_CLEAR: Predicted CLEAR images by the model.

    Returns:
        mPSNR: mean PSNR between the real CLEAR images and predicted CLEAR images.
    """
    with tf.name_scope('psnr'):
        MSE = tf.reduce_mean(tf.square(CLEAR - pred_CLEAR), [1, 2, 3])
        T = tf.div(1.0, MSE) #除法
        R = 10.0 * tf.div(tf.log(T), tf.log(10.0))
        mPSNR = tf.reduce_mean(R, name = 'mean_psnr')
        
    # Attach a scalar summary to mPSNR
    tf.summary.scalar(mPSNR.op.name, mPSNR)
        
    return mPSNR



def Dics(img_label_batch, img_seg_batch):
    '''
   Compute the dics  coefficient between the label images and the predicted label images.
   Before dics is a Threshold operation.
    
   Args:
       img_label_batch: ground truth label images.
       img_seg_batch: predicted label images.
       
   Returns: dics coefficient.
    '''
    num_label = 0
    num_seg = 0
    num_cover = 0
    dics2 = 0
    
    for j in range(img_label_batch.shape[0]):
        
        for i in range(img_label_batch.shape[1]):
        
            for k in range(img_seg_batch.shape[2]):
                
                img_label_batch[j, i, k, :] = img_label_batch[j, i, k, :] * 255 
                img_seg_batch[j, i, k, :] = img_seg_batch[j, i, k, :] * 255
                x = img_label_batch[j, i, k, :]
                y = img_seg_batch[j, i, k, :]
                if x == 255:
                    num_label = num_label + 1
                if y == 255:
                    num_seg = num_seg + 1
                if (x == 255) and (y == 255):
                   num_cover = num_cover + 1
        dics1 = 2*num_cover / (num_label + num_seg)        
        dics2 += dics1
    dics = dics2 / (img_label_batch.shape[0])
    return dics













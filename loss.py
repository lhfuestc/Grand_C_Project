# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 18:51:35 2018

@author: zxlation
"""
import tensorflow as tf
import options
import numpy as np
TOTAL_LOSS_COLLECTION = options.TOTAL_LOSS_COLLECTION

def calc_batch_tv(batch_image):
    row = batch_image[:, 1:, :, :] - batch_image[:, :-1, :, :]
    col = batch_image[:, :, 1:, :] - batch_image[:, :, :-1, :]
    row = tf.abs(row)
    col = tf.abs(col)
    
    return (tf.reduce_mean(row) + tf.reduce_mean(col))/2.0


def loss_l2_ds(real_HR, pred_HR, DS_1, DS_2):
    """Calculates the L2 loss from the real HR images and the predicted HR images.
            total_loss = ||real_HR - pred_HR||_L2 + η·WD
                       = ||real_HR - pred_HR||_2^2 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR

    Returns:
        total_loss: MSE between real HR images and predicted HR images and weight decay(optional).
    """
    with tf.name_scope('l2_loss'):
        mse_loss = tf.losses.mean_squared_error(real_HR, pred_HR)
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, mse_loss)
        mse_loss_ds = tf.losses.mean_squared_error(DS_1, DS_2)
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, mse_loss_ds*0.01)
        
        # Attach a scalar summary to mse_loss
        tf.summary.scalar(mse_loss.op.name, mse_loss)
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION), name = 'total_loss_l2')


def loss_l2(real_HR, pred_HR):
    """Calculates the L2 loss from the real HR images and the predicted HR images.
            total_loss = ||real_HR - pred_HR||_L2 + η·WD
                       = ||real_HR - pred_HR||_2^2 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR

    Returns:
        total_loss: MSE between real HR images and predicted HR images and weight decay(optional).
    """
    with tf.name_scope('l2_loss'):
        mse_loss = tf.losses.mean_squared_error(real_HR, pred_HR)
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, mse_loss)
        
        # Attach a scalar summary to mse_loss
        tf.summary.scalar(mse_loss.op.name, mse_loss)
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION), name = 'total_loss_l2')


def loss_l2_dice(real_HR, pred_HR, dice_wt):
    """Calculates the L2 loss from the real HR images and the predicted HR images.
            total_loss = ||real_HR - pred_HR||_L2 + η·dice
                       = ||real_HR - pred_HR||_2^2 + η.dice
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR

    Returns:
        total_loss: MSE between real HR images and predicted HR images and weight decay(optional).
    """
    with tf.name_scope('l2_loss_dice'):
        
        scalar_tensor = tf.constant(value=0.0001)
        intersection = tf.reduce_sum(real_HR * pred_HR)
        sum_real = tf.add(tf.reduce_sum(real_HR),scalar_tensor)
        sum_pred = tf.reduce_sum(pred_HR)
        dice_loss = 1 - (2.0 * intersection + 0.0001 ) / (sum_real + sum_pred + 0.0001)
#        dice_loss = tf.subtract(1.0, (tf.add(tf.multiply(2.0, intersection), scalar_tensor)) / tf.add(sum_real, sum_pred))
        mse_loss = tf.losses.mean_squared_error(real_HR, pred_HR)
        l2_dice_loss = tf.add(mse_loss, dice_wt*dice_loss) 
        
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, l2_dice_loss)
        
        # Attach a scalar summary to mse_loss
        tf.summary.scalar("l2_dice_loss", l2_dice_loss)
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION), name = 'total_loss_l2')


def loss_l1(real_HR, pred_HR):
    """Calculates the L1 loss from the real HR images and the predicted HR images.
              total_loss = ||real_HR - pred_HR||_L1 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR
    Returns:
        total loss: L1 loss between real clear images and predicted HR images.
    """
    with tf.name_scope('l1_loss'):
        abs_loss = tf.reduce_mean(tf.abs(real_HR - pred_HR))
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, abs_loss)
        tf.summary.scalar(abs_loss.op.name, abs_loss)
        
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))



def loss_lp(real_HR, pred_HR, p):
    """Calculates the Lp loss from the real HR images and the predicted HR images.
              total_loss = ||real_HR - pred_HR||_Lp + η·WD
       where 'WD' indicates weight decay. p should be smaller than 1.0
       --> try: (2/3)*[x^(3/2)]
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR
        p: the factor of the loss
    Returns:
        total loss: Lp loss between real clear images and predicted HR images.
    """
    
    alpha = 1e-2
    with tf.name_scope('lp_loss'):
        lp_loss = tf.reduce_mean(tf.pow(tf.abs(real_HR - pred_HR) + alpha, p))
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, lp_loss)
        tf.summary.scalar(lp_loss.op.name, lp_loss)
        
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))



def weighted_loss_l2(real_HR, pred_HR, weights = [1, 1]):
    """Calculates the L2 loss from the real HR images and the predicted HR images.
            total_loss = ||real_HR - pred_HR||_L2 + η·WD
                       = ||real_HR - pred_HR||_2^2 + η·WD
       where 'WD' indicates weight decay.
    Args:
        real_HR: The ground-truth HR images with shape of [batch_size, height, width, channels]
        pred_HR: The predicated HR images by model with the same shape as real_HR

    Returns:
        total_loss: MSE between real HR images and predicted HR images and weight decay(optional).
    """
    with tf.name_scope('l2_loss'):
        weight_map = tf.to_float(tf.equal(real_HR, 0.0, name='label_map_0')) * weights[0]
        
        for i, weight in enumerate(weights[1:], start=1):
            weight_map = weight_map + tf.to_float(tf.equal(real_HR, i, name='label_map_' + str(i))) * weight

        weight_map = tf.stop_gradient(weight_map, name='stop_gradient')
    
        mse_loss = tf.reduce_mean(tf.square(real_HR - pred_HR) * weight_map)
        
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, mse_loss)
        
        # Attach a scalar summary to mse_loss
        tf.summary.scalar(mse_loss.op.name, mse_loss)
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION), name = 'total_loss_l2')


def weighted_softmax_cross_entropy_loss(logits, labels, weights):
    """
    Computes the SoftMax Cross Entropy loss with class weights based on the class of each pixel.

    Parameters
    ----------
    logits: TF tensor
        The network output before SoftMax.
    labels: TF tensor
        The desired output from the ground truth.
    weights : list of floats
        A list of the weights associated with the different labels in the ground truth.

    Returns
    -------
    loss : TF float
        The loss.
    weight_map: TF Tensor
        The loss weights assigned to each pixel. Same dimensions as the labels.
    
    """

    with tf.name_scope('loss'):
        labels = tf.cast(labels, tf.int32)
#        tf.clip_by_value(logits,1e-8,1.0)
#        logits = tf.reshape(logits, [-1], name='flatten_logits')
        logits = tf.reshape(logits, [-1, tf.shape(logits)[3]], name='flatten_logits')
##        logits = tf.reshape(logits, [-1], name='flatten_logits')
        labels = tf.reshape(labels, [-1], name='flatten_labels')
#
#        weight_map = tf.to_float(tf.equal(labels, 0, name='label_map_0')) * weights[0]
#        for i, weight in enumerate(weights[1:], start=1):
#            weight_map = weight_map + tf.to_float(tf.equal(labels, i, name='label_map_' + str(i))) * weight
#
#        weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

        # compute cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_softmax')

        # apply weights to cross entropy loss
        #weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

        # get loss scalar
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

    return loss



def huber_loss(real_WH, pred_WL, delta):
    
    with tf.name_scope('huber_loss'):
        huber_loss = tf.losses.huber_loss(real_WH, pred_WL, delta)
        tf.add_to_collection(TOTAL_LOSS_COLLECTION, huber_loss)
        tf.summary.scalar(huber_loss.op.name, huber_loss)
    return tf.add_n(tf.get_collection(TOTAL_LOSS_COLLECTION))   
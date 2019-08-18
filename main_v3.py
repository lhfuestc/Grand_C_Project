# -*- coding: utf-8 -*-
"""
Created on Wed Des 25 16:23:37 2018

@author: LHF
"""

from __future__ import absolute_import
from skimage import measure
from datetime import datetime
from models import model_2d
from scipy import misc


import tensorflow as tf
import numpy as np
import termcolor
import optimize
import options
import loss
import time
import cv2
import os

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

norm = 255.0

#training settings
learning_rate = options.params.learning_rate
batch_size = options.params.batch_size
max_steps = options.params.max_steps
image_H = options.img_H
image_W = options.img_W
label_H = options.lab_H
label_W = options.lab_W
inpRow = options.InpRow
inpCol = options.InpRow
epoch = 10000
# log settings
train_log_freq = options.params.train_log_freq
train_log_dir = options.params.train_log_dir

# retraining settings
train_from_exist = options.params.train_from_exist
exist_model_dir = options.params.exist_model_dir

# dataset directory
train_seg_dir = options.params.train_seg_dir
train_lab_dir = options.params.train_lab_dir
valid_seg_dir = options.params.valid_seg_dir
valid_lab_dir = options.params.valid_lab_dir

# validating settings
valid_save_images = options.params.valid_save_image
valid_image_save_dir = options.params.valid_image_save_dir
valid_save_dir = options.params.valid_log_dir

def DC_TPR_TNF(pred, label, index):
    
    '''
       calculate:
       the dics(DIC: 正确率)
       sensitivity(TPR:true positive rate，描述识别出的所有正例占所有正例的比例 ) 
       specificity(TNR:true negative rate，描述识别出的负例占所有负例的比例) 
       between img_lab and img_seg.     
    Args:
        - img_lab:
            The groundtruth label image.
        - img_seg:
            The produced label image.
        - TN -> label:false,seg:false; FP -> label:false,seg:true; FN -> label:true,seg:false; TP -> label:true,seg:true.
    returns:
        mean_dic mean_tpr and mean_tnr.
    '''
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    mean_dic = 0
    mean_tpr = 0
    mean_tnr = 0
    for k in range(batch_size):
        pred_lab, real_lab = pred[k, :, :, 0], label[k, :, :, 0]
        for i in range(pred_lab.shape[0]):
            for j in range(real_lab.shape[1]):
                x = pred_lab[i, j]
                y = real_lab[i, j]
                if (x in index) and (y in index):
                    TN = TN + 1
                if (x in index) and (y not in index):
                    FP = FP + 1
                if (x not in index) and (y in index):
                    FN = FN + 1
                if (x not in index) and (y not in index):
                    TP = TP + 1
        TPR = TP / (TP + FN + 0.001)
        TNR = TN / (TN + FP + 0.001)
        DIC = 2*TN / (2*TN + FN + FP +0.001) 
        mean_dic += DIC
        mean_tpr += TPR
        mean_tnr += TNR
    LIST = mean_dic, mean_tpr, mean_tnr          
    return LIST
   
def raw_dice_measurements(logits, ground_truth, label=1):
    """
    Computes the sum of pixels of the given label of interest in the prediction, in the the ground truth and
    in the intersection of the two. These values can be used to compute the Dice score.

    Parameters
    ----------
    logits: TF tensor
        The network output before SoftMax.
    ground_truth: TF tensor
        The desired output from the ground truth.
    label: int
        The label of interest.

    Returns
    -------
    sum_prediction : TF float
        The sum of pixels with the given label in the prediction.
    sum_ground_truth: TF float
        The sum of pixels with the given label in the ground truth.
    sum_intersection: TF float
        The sum of pixels with the given label in the intersection of prediction and ground truth.    
    """

    with tf.name_scope('measurements'):

        label_const = tf.constant(label, dtype=tf.int32, shape=[], name='label_of_interest')

        prediction = tf.to_int32(tf.argmax(logits, 3, name='prediction'))

        prediction_label = tf.equal(prediction, label_const)
        ground_truth_label = tf.equal(tf.to_int32(ground_truth[:,:,:,0]), label_const)

        sum_ground_truth = tf.reduce_sum(tf.to_float(ground_truth_label), name='sum_ground_truth')
        sum_prediction = tf.reduce_sum(tf.to_float(prediction_label), name='sum_prediction')

        with tf.name_scope('intersection'):
            sum_intersection = tf.reduce_sum(tf.to_float(tf.logical_and(prediction_label, ground_truth_label)))
    
    return (sum_prediction, sum_ground_truth, sum_intersection) 

  
def calc_dics(pred, label):
    mean_dics = 0
    num_example = pred.shape[0]
    for i in range(num_example):
        pred_lab, real_lab = pred[i, :, :, 0], label[i, :, :, 0]
        pred_lab = np.ceil(pred_lab) 
        real_lab = np.ceil(real_lab)
        
        intersection = np.sum(pred_lab*real_lab)
        summation_1 = np.sum(pred_lab)
        summation_2 = np.sum(real_lab)
        
        loss_dics = (2.0 * intersection + 0.0001 ) / (summation_1 + summation_2 + 0.0001)
        mean_dics += loss_dics
    mean_dics = mean_dics / num_example
    return mean_dics


def HSDF(img_lab, img_seg, index):
    ''' calculate the distance of hausdorff.
     Args:
        - img_lab:
            The groundtruth label image.
        - img_seg:
            The produced label image.
    Returns:
        HSDF
    '''
       
    num = len(index)
    for i in range(num):
        bool_lab = img_lab == index[i]
        img_lab[bool_lab] = 3
        bool_seg = img_seg == index[i]
        img_seg[bool_seg] = 3
        
    bool_lab = img_lab != 3
    bool_seg = img_seg != 3
    img_lab[bool_lab] = 0
    img_seg[bool_seg] = 0
    contours_lab = measure.find_contours(img_lab, 0.5)
    contours_seg = measure.find_contours(img_seg, 0.5)
    contour_labs = contours_lab[0]
    contour_segs = contours_seg[0]
    m = contour_labs.shape[0]
    n = contour_segs.shape[0]
    dist_lab = np.array([m, 1], dtype = np.float32)
    dist_seg = np.array([n, 1], dtype = np.flaot32)
    for j in range(m):
        dist_lab[j, :] = np.min((contour_segs - contour_labs[j, :])**2,axis=1)
    
    for k in range(n):
        dist_seg[k, :] = np.min((contour_labs - contour_segs[j, :])**2,axis=1)
    
    HSDF = np.max([np.max(dist_lab), np.max(dist_seg)])
    
    return HSDF
        
    
def calc_psnr_and_ssim(real_batch, pred_batch):
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
#    pred_batch = pred_batch[:,:,:,0] + pred_batch[:,:,:,1]
#    pred_batch = pred_batch[:,:,:,np.newaxis]
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
        #seg_im = (np.load(seg_dir + seg_names[i]) / norm_16).astype(np.float32)
        seg_im = (misc.imread(seg_dir + seg_names[i]) / norm).astype(np.float32)
        #seg_im = transform.resize(seg_im, [256, 256], order = 3, mode = 'constant', cval = 0)
        lab_im = (misc.imread(lab_dir + lab_names[i]) / norm).astype(np.float32)
        #lab_im = transform.resize(lab_im, [256, 256], order = 3, mode = 'constant', cval = 0)
        seg_im = seg_im[:, :, np.newaxis]
        lab_im = lab_im[:, :, np.newaxis]
        seg_images.append(seg_im)
        lab_images.append(lab_im)
    return seg_images, lab_images


def pps_pipeline(org_image):
    """
    Creates the pre-processing pipeline used during training. 
    
    Parameters
    ----------
    org_image: original image ---->>> 16 bit numpy.
        The image will be introduced into the model before training.
    Returns
    -------
    std_image: standard image
        standard image after a series of processing
    """
    
    org_image = (org_image - np.min(org_image)) / (np.max(org_image) - np.min(org_image))
    org_image = np.uint8(org_image*255)
    
    std_image = org_image.copy()
    _,binary = cv2.threshold(org_image,100,1,cv2.THRESH_BINARY)  
    
    image,contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(contour) for contour in contours]
    
    max_area_ind = areas.index(max(areas))
    mask = np.zeros_like(std_image)
    
    cv2.fillPoly(mask, [contours[max_area_ind]], 255)
    std_image = cv2.bitwise_and(std_image, std_image, mask = mask)
    
    std_image  = cv2.equalizeHist(std_image)
    return std_image


def read_images_16(seg_dir, lab_dir):
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
            errstr = "seg image name %s does not match lab image name .load%s!" % (seg_name, lab_name)
            raise ValueError(errstr)
        # read seg & lab image pair
#        seg_img = np.load(seg_dir + seg_names[i])
#        seg_img = pps_pipeline(seg_img)
#        seg_im = ((seg_img - seg_img.min()) / (seg_img.max() - seg_img.min())).astype(np.float32)
#        seg_im = (np.load(seg_dir + seg_names[i]) / norm_16).astype(np.float32)
        seg_im = (misc.imread(seg_dir + seg_names[i]) / norm).astype(np.float32)
        #seg_im = transform.resize(seg_im, [256, 256], order = 3, mode = 'constant', cval = 0)
#        seg_im = ((seg_img - np.min(seg_img)) / (np.max(seg_img) - np.min(seg_img))).astype(np.float32)
        lab_im = (misc.imread(lab_dir + lab_names[i]) / norm).astype(np.float32)
#        lab_im = ((lab_img - np.min(lab_img)) / (np.max(lab_img) - np.min(lab_img))).astype(np.float32)
#        seg_im = (seg_im - np.mean(seg_im)) / np.std(seg_im)
        
        #lab_im = transform.resize(lab_im, [256, 256], order = 3, mode = 'constant', cval = 0)
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
    for i in range(batch_size):
        # generate an image patch randomly
        index = np.random.randint(low = 0, high = num_img)
        seg_image = seg_images[index]
        lab_image = lab_images[index]
#        center_h = int(seg_image.shape[0] / 2)
#        center_w = int(seg_image.shape[1] / 2)
#        seg_patch = seg_image[center_h - int(inpRow/2):center_h + int(inpRow/2), center_w - int(inpCol/2):center_w + int(inpCol/2), :]
#        lab_patch = lab_image[center_h - int(inpRow/2):center_h + int(inpRow/2), center_w - int(inpCol/2):center_w + int(inpCol/2), :]
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


def restore_model(sess, saver, exist_model_dir, global_step):
    log_info = "Restoring Model From %s..." % exist_model_dir
    print(termcolor.colored(log_info, 'green', attrs = ['bold']))
    ckpt = tf.train.get_checkpoint_state(exist_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, init_step))
    else:
        print('No Checkpoint File Found!')
        return
    
    return init_step


def chop_forward(inputs, sess, lr_image, hr_model, shave = 64, chopSize = 400*400):
    B, H, W, C = inputs.shape
    hc, wc = int(np.ceil(H/2)), int(np.ceil(W/2))
    
    inpPatch = [inputs[:, 0:hc + shave, 0:wc + shave, :],
                inputs[:, 0:hc + shave, wc - shave:W, :],
                inputs[:, hc - shave:H, 0:wc + shave, :],
                inputs[:, hc - shave:H, wc - shave:W, :]]
    
    outPatch = []
    if chopSize > (wc * hc):
        for i in range(4):
            out_batch = sess.run(hr_model, feed_dict = {lr_image:inpPatch[i]})
            outPatch.append(out_batch)
    else:
        for i in range(4):
            out_batch = chop_forward(inpPatch[i], sess, lr_image, hr_model, shave, chopSize)
            outPatch.append(out_batch)
    
    ret = np.zeros([B, H, W, 1], dtype = np.float32)
    
    ret[:, 0:hc, 0:wc, :] = outPatch[0][:, 0:hc, 0:wc, :]
    ret[:, 0:hc, wc:W, :] = outPatch[1][:, 0:hc, shave:, :]
    ret[:, hc:H, 0:wc, :] = outPatch[2][:, shave:, 0:wc, :]
    ret[:, hc:H, wc:W, :] = outPatch[3][:, shave:, shave:, :]
    
    return ret


def train(): 
    """Train SR model for MAX_STEPS steps."""
    global_step = tf.train.get_or_create_global_step()
    
    seg_batch = tf.placeholder(dtype = tf.float32, shape = [None, inpRow, inpCol, 1])
    
    lab_batch = tf.placeholder(dtype = tf.float32, shape = [None, inpRow, inpCol, 1])
    
    tf.summary.image('seg_image', seg_batch, max_outputs = 4)
    
    tf.summary.image('lab_image', lab_batch, max_outputs = 4)

    # build a computational graph that computes the predicted clear images from gibbs images
    print(termcolor.colored("Building Computation Graph...", 'green', attrs = ['bold']))
    
    pred_model = model_2d.Dense_Unet_CA(seg_batch)
    # Calculate loss
#    train_loss = loss.weighted_loss_l2(pred_model, lab_batch, [1, 1])
#    raw_dice = raw_dice_measurements(pred_model, lab_batch)
#    train_loss = loss.weighted_softmax_cross_entropy_loss(pred_model, lab_batch,[1,1])
    train_loss = loss.loss_l2_dice(pred_model, lab_batch, 0.1)
#    train_loss = loss.loss_l2(pred_model, lab_batch)
    # Get the training op for optimizing loss
    train_op = optimize.optimize(train_loss, learning_rate, global_step)
    
    # Create a saver.
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 3)
    
    # Build the summary operation from the last tower summaries.
    summ_op = tf.summary.merge_all()
    
    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU implementations.
    config = tf.ConfigProto()
    
    config.log_device_placement = True
    
    config.allow_soft_placement = True
    
    sess = tf.Session(config = config)
    
    # retrain the existed models
    init_step = 0
    if train_from_exist:
        init_step = restore_model(sess, saver, exist_model_dir, global_step)
    else:
        print(termcolor.colored("Initializing Variables...", 'green', attrs = ['bold']))
        sess.run(tf.global_variables_initializer())
    
    # define summary writer
    print(termcolor.colored("Defining Summary Writer...", 'green', attrs = ['bold']))
    summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        
    min_loss = float('Inf')
    
    max_dics = float('-Inf')
    
    # get training images
    train_seg_images, train_lab_images = read_images(train_seg_dir, train_lab_dir)
   
    # get valid images
    valid_seg_images, valid_lab_images = read_images(valid_seg_dir, valid_lab_dir)
    
    print(termcolor.colored("Starting To Train...", 'green', attrs = ['bold']))
    
    for step in range(init_step, max_steps):
        step += 1
        start_time = time.time()
        inpBatch, labBatch = get_train_batch(train_seg_images, train_lab_images)
        feed_dict = {seg_batch:inpBatch, lab_batch:labBatch}
        sess.run(train_op, feed_dict = feed_dict)
        duration = time.time() - start_time
        if (step + 1) % train_log_freq == 0:
            examples_per_second = batch_size/duration
            seconds_per_batch = float(duration)
                    
            run_list = [train_loss, pred_model]
            model_loss, model_lab = sess.run(run_list, feed_dict = feed_dict)
#            train_raw_dice = sess.run(raw_dice_measurements(model_lab, labBatch))
            
   #         model_dics = (2*train_raw_dice[2] + 0.00001)/(train_raw_dice[0] + train_raw_dice[1] + 0.00001)
            model_dics = calc_dics(model_lab, labBatch)
            
            if model_loss < min_loss: min_loss = model_loss
            
            if model_dics > max_dics: max_dics = model_dics
            print(termcolor.colored('%s ------ step #%d' % (datetime.now(), step + 1), 'green', attrs = ['bold']))
            print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
            print('  DICS = %.6f\t MAX_DICS = %.6f' % (model_dics, max_dics))
            print('  ' + termcolor.colored('%.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch), 'blue', attrs = ['bold']))         
            
           
    summary_writer.close()
    sess.close()
    

def main(argv = None):  # pylint: disable = unused - argument
    if not train_from_exist:
        if tf.gfile.Exists(train_log_dir):
            tf.gfile.DeleteRecursively(train_log_dir)
        tf.gfile.MakeDirs(train_log_dir)
    else:
        if not tf.gfile.Exists(exist_model_dir):
            raise ValueError("Train from existed model, but the target dir does not exist.")
        
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()

    




    
    
    
    
    
    
    

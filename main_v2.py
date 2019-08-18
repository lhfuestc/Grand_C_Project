# -*- coding: utf-8 -*-
"""
Created on Wed Des 25 16:23:37 2018

@author: LHF
"""

from __future__ import absolute_import
from skimage import measure, transform
from datetime import datetime
from models import model_2d
from scipy import misc
import scipy.ndimage

import numbers
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


def random_translation(seg_img, lab_img, probability=1.0, border_usage=0.5, default_border=0.25, label_of_interest=None, default_pixel=None, default_label=None):
    """
    Translates a random selection of the input by a random amount. The translation varies between datapoints, but is the same for
    all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    probability: float
        The probability of translating the input. If it is below 1, some inputs will be passed through unchanged.
    border_usage: float
        If a label_of_interest is set, the border is defined as the area between the rectangular bounding box around the region of
        interest and the edge of the image. This parameter defines how much of that border may be used for translation or, in other
        words, how much of that border may end up outside of the new image.
    default_border: float
        The amount of translation that is possible with respect to the input size, if label_of_interest is either not specified or
        not found in the input. border_usage does not apply to the default_border.
    label_of_interest: int
        The label of interest in the ground truth.
    default_pixel: number or None
        The fill value for the image input. If None, the minimum value will be used.
    default_label: number or None
        The fill value for the ground truth input. If None, the minimum value will be used.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(border_usage, numbers.Number):
        raise TypeError("Border usage must be a number! Received: {}".format(type(border_usage)))

    if not isinstance(default_border, numbers.Number):
        raise TypeError("Default border must be a number! Received: {}".format(type(default_border)))

    if label_of_interest is not None and not isinstance(label_of_interest, numbers.Number):
        raise TypeError("Label of interest must be a number! Received: {}".format(type(label_of_interest)))

    if default_pixel is not None and not isinstance(default_pixel, numbers.Number):
        raise TypeError("Default pixel must be a number! Received: {}".format(type(default_pixel)))

    if default_label is not None and not isinstance(default_label, numbers.Number):
        raise TypeError("Default label must be a number! Received: {}".format(type(default_label)))

    if probability > 1.0 or probability < 0.0:
        raise ValueError("Probability must be between 0.0 and 1.0! Received: {}".format(probability))

    if border_usage > 1.0 or border_usage < 0.0:
        raise ValueError("Border usage must be between 0.0 and 1.0! Received: {}".format(border_usage))

    if default_border > 1.0 or default_border < 0.0:
        raise ValueError("Default border must be between 0.0 and 1.0! Received: {}".format(default_border))

    def non(s):
        return s if s < 0 else None

    def mom(s):
        return max(0, s)


    if(np.random.rand() < probability):

        if label_of_interest is None or label_of_interest not in lab_img:
            xdist = default_border * lab_img.shape[0]
            ydist = default_border * lab_img.shape[1]

        else:
            itemindex = np.where(lab_img == label_of_interest)

            xdist = min(np.min(itemindex[0]), lab_img.shape[0] - np.max(itemindex[0])) * border_usage
            ydist = min(np.min(itemindex[1]), lab_img.shape[1] - np.max(itemindex[1])) * border_usage

        ox = np.random.randint(-xdist, xdist) if xdist >= 1 else 0
        oy = np.random.randint(-ydist, ydist) if ydist >= 1 else 0

        fill_value = default_pixel if default_pixel is not None else np.min(seg_img)
        shift_seg = np.full_like(seg_img, fill_value)
        shift_seg[mom(ox):non(ox), mom(oy):non(oy), ...] = seg_img[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]

        fill_value = default_label if default_label is not None else np.min(lab_img)
        shift_lab = np.full_like(lab_img, fill_value)
        shift_lab[mom(ox):non(ox), mom(oy):non(oy), ...] = lab_img[mom(-ox):non(-ox), mom(-oy):non(-oy), ...]
    else:
        shift_seg = seg_img
        shift_lab = lab_img
    return shift_seg, shift_lab


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
        seg_im = transform.resize(seg_im, [256, 256], order = 1, mode = 'constant', cval = 0)
        lab_im = (misc.imread(lab_dir + lab_names[i]) / norm).astype(np.float32)
        lab_im = transform.resize(lab_im, [256, 256], order = 0, mode = 'constant', cval = 0)
        seg_im = seg_im[:, :, np.newaxis]
        lab_im = lab_im[:, :, np.newaxis]
        seg_images.append(seg_im)
        lab_images.append(lab_im)
    return seg_images, lab_images


def random_rotation(seg_img, lab_img, probability=1.0, upper_bound=180):
    """
    Rotates a random selection of the input by a random amount. The rotation varies between datapoints, but is the same for
    all inputs of a single datapoint.

    Parameters
    ----------
    source : iterable
        An iterable over a number of datapoints where each datapoint is a tuple of a list of inputs and a parameter dictionary.
    probability: float
        The probability of rotating the input. If it is below 1, some inputs will be passed through unchanged.
    upper_bound: number
        The maximum rotation in degrees.

    Returns
    -------
    gen : generator
        A generator that yields each transformed datapoint as a tuple of a list of inputs and a parameter dictionary.
    """
    if not isinstance(probability, numbers.Number):
        raise TypeError("Probability must be a number! Received: {}".format(type(probability)))

    if not isinstance(upper_bound, numbers.Number):
        raise TypeError("Upper bound must be a number! Received: {}".format(type(upper_bound)))

    if upper_bound < 0:
        raise ValueError("Upper bound must be greater than 0! Received: {}".format(upper_bound))
    elif upper_bound > 180:
        upper_bound = 180


    if(np.random.rand() < probability):

        angle = np.random.randint(-upper_bound, upper_bound)
        angle = (360 + angle) % 360
        seg_img = scipy.ndimage.interpolation.rotate(seg_img, angle, reshape=False, order=1, cval=np.min(seg_img), prefilter=False)  # order = 1 => biliniear interpolation
        lab_img = scipy.ndimage.interpolation.rotate(lab_img, angle, reshape=False, order=0, cval=np.min(lab_img), prefilter=False)  # order = 0 => nearest neighbour
    return seg_img, lab_img


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
        
        # do some augmentation here 
        seg_patch, lab_patch = random_rotation(seg_patch, lab_patch, probability=0.9, upper_bound=180)
        #seg_patch, lab_patch = random_translation(seg_patch, lab_patch, probability=0.9, border_usage=0.8, default_border=0.25, label_of_interest=1.0, default_pixel=None, default_label=None)    
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
#    train_loss = loss.loss_l2_dice(pred_model, lab_batch, 0.5)
    train_loss = loss.loss_l2(pred_model, lab_batch)
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
    
    max_psnr = float('-Inf')
    
    max_ssim = float('-Inf')
    
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
            model_psnr, model_ssim = calc_psnr_and_ssim(labBatch, model_lab)

#            model_dics = (2*train_raw_dice[2] + 0.00001)/(train_raw_dice[0] + train_raw_dice[1] + 0.00001)
            model_dics = calc_dics(model_lab, labBatch)
            
            if model_loss < min_loss: min_loss = model_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            if model_psnr > max_psnr: max_psnr = model_psnr
            if model_ssim > max_ssim: max_ssim = model_ssim
            if model_dics > max_dics: max_dics = model_dics
            print(termcolor.colored('%s ------ step #%d' % (datetime.now(), step + 1), 'green', attrs = ['bold']))
            print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
            print('  PSNR = %.6f\t MAX_PSNR = %.6f' % (model_psnr, max_psnr))
            print('  SSIM = %.6f\t MAX_SSIM = %.6f' % (model_ssim, max_ssim))
            print('  DICS = %.6f\t MAX_DICS = %.6f' % (model_dics, max_dics))
            print('  ' + termcolor.colored('%.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch), 'blue', attrs = ['bold']))         

                  
        if ((step + 1) % 200 == 0) or ((step + 1) == max_steps):
            summary_str = sess.run(summ_op, feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, step + 1)
                    
        if (step == 0) or ((step + 1) % 1000 == 0) or ((step + 1) == max_steps):
            checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
            print("saving checkpoint into %s-%d" % (checkpoint_path, step + 1))
            saver.save(sess, checkpoint_path, global_step = step + 1)
           
        if (step == 0) or ((step + 1) % 500 == 0) or ((step + 1) == max_steps):
            valid_input_batch, valid_label_batch = get_valid_batch(valid_seg_images, valid_lab_images)
            #valid_model = chop_forward(valid_input_batch, sess, seg_batch, pred_model)
            valid_model = sess.run(pred_model, feed_dict = {seg_batch:valid_input_batch})
            #valid_model = np.ceil(valid_model)
            valid_loss = np.mean(np.abs(valid_label_batch - valid_model))
            # calculating on valid images
            valid_model_psnr, valid_model_ssim = calc_psnr_and_ssim(valid_label_batch, valid_model)
           # mean_wt, mean_et, mean_tc = calc_WT_ET_and_TC(valid_label_batch, valid_model, valid_batch_size)
            valid_dics = calc_dics(valid_model, valid_label_batch)
           # valid_raw_dice = sess.run(raw_dice_measurements(valid_model, valid_label_batch))
            #valid_dics = (2*valid_raw_dice[2] + 0.00001)/(valid_raw_dice[0] + valid_raw_dice[1] + 0.00001)
            with open(valid_save_dir + "valid_records.txt", "a") as file:
                format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
                file.write(str(format_str) % (step + 1, valid_loss, valid_model_psnr, valid_model_ssim, valid_dics))
              
            if valid_save_images:
                index_valid = np.random.randint(low = 0, high = 30)
                image_path = os.path.join(valid_image_save_dir, str(step + 1) + '_label.png')
                valid_label_img = valid_label_batch[index_valid, :, :, 0]
                misc.imsave(image_path, valid_label_img*norm)
                    
                image_path = os.path.join(valid_image_save_dir, str(step + 1) + '_model.png')
                valid_model_img = valid_model[index_valid, :, :, 0]
                misc.imsave(image_path, valid_model_img*norm)
           
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

    




    
    
    
    
    
    
    

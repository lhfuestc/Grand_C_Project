# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:07:39 2017

@author: zxlation
"""

from __future__ import absolute_import
from scipy import misc
from datetime import datetime
from skimage import measure
from models import model_2d
import tensorflow as tf
import numpy as np
import options
import termcolor
import time
import os

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parse command line parameters
image_H = options.img_H
image_W = options.img_W
label_H = options.lab_H
label_W = options.lab_W
inpRow = options.InpRow
inpCol = options.InpRow
nChann = options.CHANNELS
#nScale = options.SCALE
norm = 255.0

test_seg_dir = options.params.test_seg_dir
test_lab_dir = options.params.test_lab_dir
batch_size = options.params.batch_size

test_model_dir = options.params.test_model_dir
test_log_dir = options.params.test_log_dir
test_image_save_dir = options.params.test_image_save_dir
test_save_image = options.params.test_save_image


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
        
        loss_dics = (2.0 * intersection + 0.0001) / (summation_1 + summation_2 + 0.0001)
        mean_dics += loss_dics
    mean_dics = mean_dics / num_example
    return mean_dics


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
        seg_im = (misc.imread(seg_dir + seg_names[i]) / norm).astype(np.float32)
        #seg_im = transform.resize(seg_im, [256, 256], order = 3, mode = 'constant', cval = 0)
        lab_im = (misc.imread(lab_dir + lab_names[i]) / norm).astype(np.float32)
        #lab_im = transform.resize(lab_im, [256, 256], order = 3, mode = 'constant', cval = 0)
        seg_im = seg_im[np.newaxis, :, :, np.newaxis]
        lab_im = lab_im[np.newaxis, :, :, np.newaxis]
        seg_images.append(seg_im)
        lab_images.append(lab_im)
    return seg_images, lab_images



def restore_model(sess, saver, model_dir):
    # Synchronous assessment!
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        raise ValueError('no checkpoint file found!')
    
    return global_step
               

def chop_forward(inputs, sess, lr_image, hr_model, shave = 10, chopSize = 400*400):
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



def test():
    # read all valid images into memory. Note: if your dataset has a very large size, do Not
    # follow this routine.
    seg_images, lab_images = read_images(test_seg_dir, test_lab_dir)
    num_images = len(seg_images)
    if num_images != len(lab_images):
         raise ValueError("The number of test_img and test_lab does not match!")
    with tf.Graph().as_default() as g:
        # define placeholders for gibbs & clear images
#        space_mask = tf.ones([1, 512, 512, 1], dtype = tf.float32, name = 'space_mask')
        inp_batch = tf.placeholder(dtype = tf.float32, shape = [1, 512, 512, nChann])
        
        # build computational graph
        pred_model = model_2d.Dense_Unet_CA(inp_batch)
        
        # define model saver
        saver = tf.train.Saver()
        
        # collect all summary ops
        summ_op = tf.summary.merge_all()
        
        # define summary writer
        summ_writer = tf.summary.FileWriter(test_log_dir, g)
        
        # main loop for evalution
        
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.allow_soft_placement = True
        sess = tf.Session(config = config)
        
        # restore model from disk
        global_step = restore_model(sess, saver, test_model_dir)
                
        model_psnr = 0.0
        model_ssim = 0.0
        model_loss = 0.0
        model_dics = 0.0
        img_index = 0
        while img_index < num_images:
            print('processing image %d/%d...' % (img_index + 1, num_images))
            seg_image = seg_images[img_index]
            lab_image = lab_images[img_index]
            # run the model
#           model_clear = chop_forward(seg_images[img_index], sess, inp_batch, pred_model)
            start_time = time.time()
            model_clear = sess.run(pred_model, feed_dict = {inp_batch:seg_image})     
            duration = float(time.time() - start_time)
            # save images
            if test_save_image:
                    image_path = os.path.join(test_image_save_dir, str(img_index+ 1) + '_label.png')
                    test_label_img = lab_image[0, :, :, 0]
                    misc.imsave(image_path, test_label_img*norm)
            
                    image_path = os.path.join(test_image_save_dir, str(img_index + 1) + '_model.png')
                    test_model_img = model_clear[0, :, :, 0]
                    misc.imsave(image_path, test_model_img*norm)
            
            mo_loss = np.mean(np.abs(lab_image - model_clear))
            mo_dics = calc_dics(model_clear, lab_image)
            mo_psnr, mo_ssim = calc_psnr_and_ssim(lab_image.astype('float32'), model_clear.astype('float32'))
            
            model_loss += mo_loss
            model_psnr += mo_psnr
            model_ssim += mo_ssim
            model_dics += mo_dics
            
            img_index += 1
                
            
        # calculate the average PSNR over the whole validation set
        model_loss = model_loss / num_images
        model_psnr = model_psnr / num_images
        model_ssim = model_ssim / num_images
        
            
        with open(test_log_dir + "test_records.txt", "a") as file:
            format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\n"
            file.write(format_str % (int(global_step), model_loss, model_psnr, model_ssim, model_dics))
                
            
        print(termcolor.colored('%s ---- step #%d' % (datetime.now(), img_index + 1), 'green', attrs = ['bold']))
        print('  Processing Image %d/%d...' % (img_index + 1, num_images))
        print('  model_loss = %.6f\n' % (model_loss))
        print('  model_psnr = %.6f\n' % (model_psnr))
        print('  model_ssim = %.6f\n' % (model_ssim))
        print('  model_dics = %.6f\n' % (model_dics))
        print('  ' + termcolor.colored('testing one image need %.6f seconds' % (duration), 'blue', attrs = ['bold']))
        
        
        # writer relative summary into val_log_dir
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summ_op))
        summary.value.add(tag='Average model PSNR over test set', simple_value = model_psnr)
        summary.value.add(tag='Average model SSIM over test set', simple_value = model_ssim)
        summary.value.add(tag='Average model DUICS over test set', simple_value = model_dics)
        summ_writer.add_summary(summary, global_step)
        
        summ_writer.close()
        sess.close()
               
        
def main(argv = None):
    test()

if __name__ == '__main__':
    tf.app.run()   
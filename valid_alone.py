# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:07:39 2017

@author: zxlation
"""

from __future__ import absolute_import
from scipy import misc
from datetime import datetime
from skimage import measure
from models import AdaptiveResNet
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
gibbsH = options.GibbsH
gibbsW = options.GibbsW
clearH = options.ClearH
clearW = options.ClearW
nChann = options.CHANNELS
nScale = options.SCALE
norm = 255.0

valid_gibbs_dir = options.params.valid_gibbs_dir
valid_clear_dir = options.params.valid_clear_dir
valid_subset_only = options.params.valid_subset_only
if valid_subset_only:
    valid_gibbs_dir = options.VALID_SUB_GIBBS_DIR
    valid_clear_dir = options.VALID_SUB_CLEAR_DIR


interval_secs = options.params.interval_secs
evalute_once  = options.params.eval_once
batch_size = 1

valid_model_dir = options.params.valid_model_dir
valid_log_dir = options.params.valid_log_dir
image_save_dir = options.params.image_save_dir
save_image = options.params.save_image

gibbs_name_list = os.listdir(valid_gibbs_dir)
num_images = len(gibbs_name_list)


def cal_psnr_and_ssim(real_batch, pred_batch):
    """calculate the average psnr and ssim over a batch of images
    Args:
        real_HR_batch: tensor with shape of [batch_size, height, width, channels]
        pred_HR_batch: tensor with shape as real_HR_batch
    Returns:
        mean_psnr & mean_ssim over this batch of images
    """
    mean_psnr = 0.0
    mean_ssim = 0.0
    for i in range(batch_size):
        real = real_batch[i,:,:,0]
        pred = pred_batch[i,:,:,0]
        mean_ssim += measure.compare_ssim(real, pred)
        mean_psnr += measure.compare_psnr(real, pred)
        
    mean_psnr = mean_psnr/batch_size
    mean_ssim = mean_ssim/batch_size
    
    return mean_psnr, mean_ssim


def read_images(gibbs_dir, clear_dir):
    gibbs_names = sorted(os.listdir(gibbs_dir))
    clear_names = sorted(os.listdir(clear_dir))
    if len(gibbs_names) != len(clear_names):
        raise ValueError("The number of LR and HR images does not match!")
    
    gibbs_images = []
    clear_images = []
    for i in range(len(gibbs_names)):
        # check whether LR&HR image names match each other
        gibbs_name = gibbs_names[i].split('.')[0].split('_')[-1]
        clear_name = clear_names[i].split('.')[0].split('_')[-1]
        if gibbs_name != clear_name:
            errstr = "LR image name %s does not match HR image name %s!" % (gibbs_name, clear_name)
            raise ValueError(errstr)
        
        # read Gibbs & Clear image pair
        gibbs_im = misc.imread(gibbs_dir + gibbs_names[i])/norm
        clear_im = misc.imread(clear_dir + clear_names[i])/norm
        
        gibbs_images.append(gibbs_im)
        clear_images.append(clear_im)
    
    return gibbs_images, clear_images


def restore_model(sess, saver, model_dir):
    # Synchronous assessment!
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        raise ValueError('no checkpoint file found!')
    
    return global_step
               

def chop_forward(inputs, sess, lr_image, hr_model, scale, shave = 10, chopSize = 400*400):
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
            out_batch = chop_forward(inpPatch[i], sess, lr_image, hr_model, scale, shave, chopSize)
            outPatch.append(out_batch)
    
    ret = np.zeros([B, H*scale, W*scale, C], dtype = np.float32)
    H, hc = scale*H, scale*hc
    W, wc = scale*W, scale*wc
    ret[:, 0:hc, 0:wc, :] = outPatch[0][:, 0:hc, 0:wc, :]
    ret[:, 0:hc, wc:W, :] = outPatch[1][:, 0:hc, shave*scale:, :]
    ret[:, hc:H, 0:wc, :] = outPatch[2][:, shave*scale:, 0:wc, :]
    ret[:, hc:H, wc:W, :] = outPatch[3][:, shave*scale:, shave*scale:, :]
    
    return ret


def evaluate():
    # read all valid images into memory. Note: if your dataset has a very large size, do Not
    # follow this routine.
    gibbs_images, clear_images = read_images(valid_gibbs_dir, valid_clear_dir)
    
    # define batch feeders
    input_batch = np.zeros([batch_size, gibbsH, gibbsW, nChann])
    label_batch = np.zeros([batch_size, clearH, clearW, nChann])
    
    with tf.Graph().as_default() as g:
        # define placeholders for gibbs & clear images
        gibbs_batch = tf.placeholder(dtype = tf.float32, shape = [None, None, None, nChann])
        #clear_batch = tf.placeholder(dtype = tf.float32, shape = [batch_size, clearH, clearW, nChann])
        
        # build computational graph
        clear_model = AdaptiveResNet.baseline_build(gibbs_batch)
        
        # define model saver
        saver = tf.train.Saver()
        
        # collect all summary ops
        summ_op = tf.summary.merge_all()
        
        # define summary writer
        summ_writer = tf.summary.FileWriter(valid_log_dir, g)
        
        # main loop for evalution
        while True:
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            sess = tf.Session(config = config)
            
            # restore model from disk
            global_step = restore_model(sess, saver, valid_model_dir)
                    
            model_psnr = 0.0
            model_ssim = 0.0
            sinc_psnr = 0.0
            sinc_ssim = 0.0
            total_loss = 0.0
            img_index = 0
            while img_index < num_images:
                print('processing image %d/%d...' % (img_index + 1, num_images))
                
                # generate batch, assume batch_size = 1
                input_batch[0, :, :, 0] = gibbs_images[img_index]
                label_batch[0, :, :, 0] = clear_images[img_index]
                
                # run the model
                model_clear = chop_forward(input_batch, sess, gibbs_batch, clear_model, scale = nScale)
                        
                # save images
                if save_image:
                    image_name = gibbs_name_list[img_index].split('.')[0]
                    image_path = os.path.join(image_save_dir, image_name + '_model.png')
                    misc.imsave(image_path, model_clear[0, :, :, 0])
                
                model_loss = np.mean(np.abs(label_batch - model_clear))
                        
                total_loss += model_loss
                mo_psnr, mo_ssim = cal_psnr_and_ssim(label_batch.astype('float32'), model_clear.astype('float32'))
                si_psnr, si_ssim = cal_psnr_and_ssim(label_batch.astype('float32'), input_batch.astype('float32'))
                model_psnr += mo_psnr
                model_ssim += mo_ssim
                sinc_psnr += si_psnr
                sinc_ssim += si_ssim
                
                img_index += 1
                    
                
            # calculate the average PSNR over the whole validation set
            total_loss = total_loss / num_images
            model_psnr = model_psnr / num_images
            model_ssim = model_ssim / num_images
            sinc_psnr = sinc_psnr / num_images
            sinc_ssim = sinc_ssim / num_images
                
                
            with open("Records/valid_records.txt", "a") as file:
                format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n"
                file.write(format_str % (int(global_step), total_loss, model_psnr, model_ssim, sinc_psnr, sinc_ssim))
                    
                
            print(termcolor.colored("%s: loss = %.6f" % (datetime.now(), total_loss), 'green', attrs = ['bold']))
            print("  model_psnr = %.4f\tsinc_psnr = %.4f" % (model_psnr, sinc_psnr))
            print("  model_ssim = %.4f\tsinc_ssim = %.4f" % (model_ssim, sinc_ssim))
            
            # writer relative summary into val_log_dir
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summ_op))
            summary.value.add(tag='Average model PSNR over validation set', simple_value = model_psnr)
            summary.value.add(tag='Average sinc PSNR over validation set', simple_value = sinc_psnr)
            summary.value.add(tag='Average model PSNR over validation set', simple_value = model_ssim)
            summary.value.add(tag='Average sinc PSNR over validation set', simple_value = sinc_ssim)
            summ_writer.add_summary(summary, global_step)
            
            summ_writer.close()
            sess.close()
            
            if evalute_once:
                break
            else:
                time.sleep(interval_secs)
               
        

def main(argv = None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(valid_log_dir):
        tf.gfile.DeleteRecursively(valid_log_dir)
    tf.gfile.MakeDirs(valid_log_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()   
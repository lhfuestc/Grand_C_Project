# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:17:57 2018

@author: LHF
"""
from datetime import datetime
import tensorflow as tf
import numpy as np
import optimize_WH
import options_WH
import termcolor
import warnings
import model_WH
import loss_WH
import random
import time
import os

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_img_dir = options_WH.params.train_image_dir
train_lab_dir = options_WH.params.train_label_dir
valid_img_dir = options_WH.params.valid_image_dir
valid_lab_dir = options_WH.params.valid_label_dir

num_exampes = 12970
EPOCH = 100000

img_H = options_WH.ImageH
img_W = options_WH.LabelW

norm = 32767
nChann = options_WH.CHANNELS

train_log_dir = options_WH.params.train_log_dir
learning_rate = options_WH.params.learning_rate
Batch_size = options_WH.params.batch_size
max_steps = options_WH.params.max_steps

train_from_exist = options_WH.params.train_from_exist
exist_model_dir = options_WH.params.exist_model_dir

train_from_exist = options_WH.params.train_from_exist
exist_model_dir = options_WH.params.exist_model_dir


def TRE(lab_batch, pred_batch, name):
    with tf.variable_scope(name):
        num = lab_batch.shape[0]
        batch_tre = np.abs(pred_batch - lab_batch) / (lab_batch + 0.0001)
        tre_W = 0.0
        tre_H = 0.0
        for i in range(num):
            tre_W += batch_tre[i, 0]
            tre_H += batch_tre[i, 1]
        mean_tre_W = tre_W / num
        mean_tre_H = tre_H / num
        tf.summary.scalar('TRE_W', mean_tre_W)
        tf.summary.scalar('TRE_H', mean_tre_H)
    return mean_tre_W, mean_tre_H


def get_file_name_list(file_dir):
    name_list = []
    for im in os.listdir(file_dir):
        path_name = os.path.join(file_dir, im)
        name_list.append(path_name)
    
    return name_list


def get_valid_batch(valid_image_dir, valid_lab_list):
    valid_names_list = os.listdir(valid_image_dir)
    num_name = len(valid_names_list)
    valid_img_batch = np.zeros([num_name, img_H, img_W, nChann], dtype = np.float32)
    valid_lab_batch = np.zeros([num_name, 2], dtype = np.float32)
    
    for i in range(num_name):
        path_img = os.path.join(valid_image_dir, valid_names_list[i])
        index = int(valid_names_list[i].split('.')[0])
        lab = valid_lab_list[index]
        if lab[0] != index:
            errstr = "image does not match label!"
            raise ValueError(errstr)
            
        img = (np.load(path_img) / norm).astype(np.float32)
        img_batch = img[:, :, np.newaxis]
        lab_batch = (lab[1:3] / norm).astype(np.float32)
        
        valid_img_batch[i, :, :, :] = img_batch
        valid_lab_batch[i, :] = lab_batch
    return valid_img_batch, valid_lab_batch


def TAE(lab_batch, pred_batch):
    num = lab_batch.shape[0]
    batch_tae = np.abs(pred_batch - lab_batch)
    tae_W = 0.0
    tae_H = 0.0
    for i in range(num):
        tae_W += batch_tae[i, 0]
        tae_H += batch_tae[i, 1]
    mean_tae_W = tae_W / num
    mean_tae_H = tae_H / num
    return mean_tae_W, mean_tae_H


def path_name_and_label_list(img_dir, lab_dir):
    path_name_list = get_file_name_list(img_dir)
    lab_list = np.load(os.path.join(lab_dir, os.listdir(lab_dir)[0]))
    Lab_List = []
    for i in range(len(path_name_list)):
        index = int(path_name_list[i].split('/')[-1].split('.')[0])
        lab = lab_list[index]
        
        if lab[0] != index:
            errstr = "image does not match label!"
            raise ValueError(errstr)
            
        Lab_List.append(str(lab[1]) + '.' + str(lab[2]))
    return path_name_list, Lab_List
        
        
def read_npy_func(path_name, lab_name):
    """Python function to load npy files.
    
    Args:
        path_name: bytes that indicate a str name of an input image path.
        lab_name: bytes that indicate a str name  of label.
    Returns:of
        a pair of image and WH
    """
    label = np.zeros([2], dtype = np.float32)
    path_name = path_name.decode()
    lab_name = lab_name.decode()
    image = np.array(np.load(path_name), np.float32)
    for i in range(2):
        label[i] = int(lab_name.split('.')[i]) 
    image = image[:, :, np.newaxis]
    return image, label


def get_batch(IMG_DIR, LAB_DIR, batch_size):
    # get images paths for all items
    Path_Name_List,  Label_List = path_name_and_label_list(IMG_DIR, LAB_DIR)
    
    # cnvert them to tf.string
    Path_Name_List = tf.cast(Path_Name_List, tf.string)
    Label_List = tf.cast(Label_List, tf.string)
    
    # a simple judgement for consistency
    if Path_Name_List.shape[0] != Label_List.shape[0]: 
        print("Dimension Error @ namelist")
        return
    
    # build a input queue for file names
    inpQueue = tf.train.slice_input_producer([Path_Name_List, Label_List],
                                             num_epochs = None,
                                             shuffle = False,
                                             capacity = 32,
                                             shared_name = None,
                                             name = 'file_name_queue')
    
    
    input_slice, label_slice = tf.py_func(read_npy_func,
                                          [inpQueue[0], inpQueue[1]],
                                          [tf.float32, tf.float32])
    
    input_slice.set_shape([img_H, img_W, 1])
    label_slice.set_shape([2])
    
    ImageBatch,LabelBatch = tf.train.shuffle_batch([input_slice, label_slice],
                                       batch_size = batch_size,
                                       num_threads = 2,
                                       capacity = 8*batch_size,
                                       min_after_dequeue = 2*batch_size,
                                       name = 'batch_queue')
    
    
    # cast to tf.float32
    with tf.name_scope('cast_to_float32'):
        ImageBatch = tf.cast(ImageBatch, tf.float32)
        LabelBatch = tf.cast(LabelBatch, tf.float32)
        
    # Normalization
    with tf.name_scope('normalization'):
        ImageBatch = ImageBatch / norm
        LabelBatch = LabelBatch / norm
    
    return ImageBatch, LabelBatch

    
def get_train_batch(batch_names_list, train_lab_list, batch_size):
    '''
    Get image batch and label batch
    '''
    train_img_batch = np.zeros([batch_size, img_H, img_W, nChann], dtype = np.float32)
    train_lab_batch = np.zeros([batch_size, 2], dtype = np.float32)
    for i in range(batch_size):
        path_img = os.path.join(train_img_dir, batch_names_list[i])
        index = int(batch_names_list[i].split('.')[0])
        lab = train_lab_list[index]
        
        if lab[0] != index:
            errstr = "image does not match label!"
            raise ValueError(errstr)
            
        img = (np.load(path_img) / norm).astype(np.float32)   
        img_batch = img[:, :, np.newaxis]
        lab_batch = (lab[1:3] / norm).astype(np.float32)
        
        train_img_batch[i, :, :, :] = img_batch
        train_lab_batch[i, :] = lab_batch
        
    return train_img_batch, train_lab_batch
 

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
        
    
def train():
    global_step = tf.train.get_or_create_global_step()
    
    Img_Batch = tf.placeholder(dtype = tf.float32, shape = [None, None, None, nChann])
    Lab_Batch = tf.placeholder(dtype = tf.float32, shape = [None, 2])
    
    print(termcolor.colored("Building Computation Graph...", 'green', attrs = ['bold']))
    pred_model = model_WH.VGG(Img_Batch)
    
    train_loss = loss_WH.loss_l2(Lab_Batch, pred_model)
    
    train_op = optimize_WH.optimize(train_loss, learning_rate, global_step)
    
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 3)
    
    summ_op = tf.summary.merge_all()
    
    config = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
    
    sess = tf.Session(config = config)
    
    print(termcolor.colored("Defining Summary Writer...", 'green', attrs = ['bold']))
    summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    
    lab_list = np.load(os.path.join(train_lab_dir, os.listdir(train_lab_dir)[0]))
    batch_train_img, batch_train_lab = get_batch(train_img_dir, train_lab_dir, Batch_size)
    
    
    min_loss = float('Inf')
    min_tre_w = float('Inf')
    min_tae_w = float('Inf')
    min_tre_h = float('Inf')
    min_tae_h = float('Inf')
    
    with tf.Session(config = config) as sess:
        init_step = 0
        if train_from_exist:
            init_step = restore_model(sess, saver, exist_model_dir, global_step)
        else:
            print(termcolor.colored("Initializing Variables...", 'green', attrs = ['bold']))
            sess.run(tf.global_variables_initializer())
               
        coord = tf.train.Coordinator() #启动线程协调器
        threads = tf.train.start_queue_runners(coord = coord) #启动队列
        try:
            while not coord.should_stop():
                print(termcolor.colored("Starting To Train...", 'green', attrs = ['bold']))
                
                for step in range(init_step, max_steps):
                    Train_Image_Batch, Train_Label_Batch = sess.run([batch_train_img, batch_train_lab])
                    start_time = time.time()
                    
                    feed_dict = {Img_Batch:Train_Image_Batch, Lab_Batch:Train_Label_Batch}
                    _, model_loss, pred_batch = sess.run([train_op, train_loss, pred_model], feed_dict = feed_dict)
                    
                    duration = time.time() - start_time
                    Train_Label_Batch = Train_Label_Batch * norm
                    pred_batch = pred_batch * norm
                    TRE_W, TRE_H = TRE(Train_Label_Batch, pred_batch, 'train')  
                    TAE_W, TAE_H = TAE(Train_Label_Batch, pred_batch)  
                    
                    '''
                    summary = tf.Summary()
                    summary.value.add(tag='TRE_W', simple_value = TRE_W)
                    summary.value.add(tag='TAE_H', simple_value = TRE_H)
                    summary_writer.add_summary(summary, step + 1)
                    '''
                    if (step + 1) % 100 == 0:
                        examples_per_second = Batch_size/duration
                        seconds_per_batch = float(duration)
                        if model_loss < min_loss: min_loss = model_loss
                        if TRE_W < min_tre_w: min_tre_w = TRE_W
                        if TRE_H < min_tre_h: min_tre_h = TRE_H
                        if TAE_W < min_tae_w: min_tae_w = TAE_W
                        if TAE_H < min_tae_h: min_tae_h = TAE_H
            
                        print(termcolor.colored('%s ---- step #%d' % (datetime.now(), step + 1), 'green', attrs = ['bold']))
                        print('  LOSS = %.6f\t MIN_LOSS = %.6f' % (model_loss, min_loss))
                        print('  TRE_W = %.6f\t MIN_TRE_W = %.6f' % (TRE_W, min_tre_w))
                        print('  TRE_H = %.6f\t MIN_TRE_H = %.6f' % (TRE_H, min_tre_h))
                        print('  TAE_W = %.6f\t MIN_TAE_W = %.6f' % (TAE_W, min_tae_w))
                        print('  TAE_H = %.6f\t MIN_TAE_H = %.6f' % (TAE_H, min_tae_h))
                        print('  ' + termcolor.colored('%.6f images/sec\t%.6f seconds/batch' % (examples_per_second, seconds_per_batch), 'blue', attrs = ['bold']))
           
                    if ((step + 1) % 200 == 0) or ((step + 1) == max_steps):
                        summary_str = sess.run(summ_op, feed_dict = feed_dict)
                        summary_writer.add_summary(summary_str, step + 1)
                    
                    if (step == 0) or ((step + 1) % 500 == 0) or ((step + 1) == max_steps):
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        print("saving checkpoint into %s-%d" % (checkpoint_path, step + 1))
                        saver.save(sess, checkpoint_path, global_step = step + 1)
                
                    if (step == 0) or ((step + 1) % 100 == 0) or ((step + 1) == max_steps):
                        valid_input_batch, valid_label_batch = get_valid_batch(valid_img_dir, lab_list)
                        valid_model = sess.run(pred_model, feed_dict = {Img_Batch:valid_input_batch})
                        valid_label_batch = valid_label_batch * norm
                        valid_model = valid_model * norm
                
                        TRE_W, TRE_H = TRE(valid_label_batch, valid_model, 'valid')  
                        TAE_W, TAE_H = TAE(valid_label_batch, valid_model)  
           
                        valid_loss = np.mean(np.abs(valid_label_batch - valid_model))
           
                        with open("records/valid_records.txt", "a") as file:
                            format_str = "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n"
                            file.write(str(format_str) % (step + 1, valid_loss, TRE_W, TRE_H, TAE_W, TAE_H))
                    
                        with open("records/valid_records_WH.txt", "a") as file:
                            format_str = "%d\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n"
                            file.write(str(format_str) % (step + 1, 'PW', valid_model[0, 0], valid_model[1, 0], valid_model[2, 0], valid_model[3, 0], valid_model[4, 0]))
                            file.write(str(format_str) % (step + 1, 'TW', valid_label_batch[0, 0], valid_label_batch[1, 0], valid_label_batch[2, 0], valid_label_batch[3, 0], valid_label_batch[4, 0]))
                            file.write(str(format_str) % (step + 1, 'PH', valid_model[0, 1], valid_model[1, 1], valid_model[2, 1], valid_model[3, 1], valid_model[4, 1]))
                            file.write(str(format_str) % (step + 1, 'TH', valid_label_batch[0, 1], valid_label_batch[1, 1], valid_label_batch[2, 1], valid_label_batch[3, 1], valid_label_batch[4, 1]))
    
        except tf.errors.OutOfRangeError:
            print("Done!")
        finally:
            coord.request_stop()
            coord.join()
            
def main(argv = None):  
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
    
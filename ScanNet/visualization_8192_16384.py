import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_DIR = os.path.dirname(ROOT_DIR)

sys.path.append(ROOT_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
import tf_util
import pc_util

import scannet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='sem_seg', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL1 = importlib.import_module(FLAGS.model+"_1")
MODEL2 = importlib.import_module(FLAGS.model+"_2") # import network module
MODEL1_2 = importlib.import_module(FLAGS.model+"_1_2")

MODEL_FILE = os.path.join(os.path.join(ROOT_DIR, 'models'), FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train12.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()

NUM_CLASSES = 21

# Shapenet official train/test split
DATA_PATH = os.path.join(DATA_ROOT_DIR,'data')
DATA_PATH = os.path.join(DATA_PATH,'scannet')
TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT*2, split='test')

color_map = [[0, 0, 0],[174, 199, 232],[152, 223, 138],[31, 119, 180],[255, 187, 120],
             [188, 189, 34],[140, 86, 75],[255, 152, 150],[255, 0, 255],[197, 176, 213],
             [148, 103, 189],[196, 156, 148],[23, 190, 207],[247, 182, 210],[219, 219, 141],
             [255, 127, 14],[158, 218, 229],[44, 160, 44],[112, 128, 144],[227, 119, 194],[82, 84, 163]]

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'a+') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], 1.0*color[0]/255, 1.0*color[1]/255, 1.0*color[2]/255))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'a+') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [9,200,248]
            elif seg[i] == 0:
                color = [1,0,0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #2*2 8192
            pointclouds_pl1, labels_pl1, smpws_pl1  = MODEL1.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl1 = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.

            # Get model and loss 
            pred1, end_points1 = MODEL1.get_model(pointclouds_pl1, is_training_pl1, NUM_CLASSES)
            loss1 = MODEL1.get_loss(pred1, labels_pl1)
            tf.summary.scalar('loss1', loss1)

            correct1 = tf.equal(tf.argmax(pred1, 2), tf.to_int64(labels_pl1))
            accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy1', accuracy1)

            #2*2 16384
            pointclouds_pl2, labels_pl2, smpws_pl2  = MODEL2.placeholder_inputs(BATCH_SIZE, NUM_POINT*2)
            is_training_pl2 = tf.placeholder(tf.bool, shape=())
            

            # Get model and loss 
            pred2, end_points2 = MODEL2.get_model(pointclouds_pl2, is_training_pl2, NUM_CLASSES)
            loss2 = MODEL2.get_loss(pred2, labels_pl2)
            tf.summary.scalar('loss2', loss2)

            correct2 = tf.equal(tf.argmax(pred2, 2), tf.to_int64(labels_pl2))
            accuracy2 = tf.reduce_sum(tf.cast(correct2, tf.float32)) / float(BATCH_SIZE*NUM_POINT*2)
            tf.summary.scalar('accuracy2', accuracy2)

            #  1  Combination  2
            choice  = MODEL1_2.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl1_2 = tf.placeholder(tf.bool, shape=())
            

            # Get model and loss 
            pred1_2, end_points1_2 = MODEL1_2.get_model(end_points1, end_points2, choice, is_training_pl1_2, NUM_CLASSES)
            loss1_2, otherloss = MODEL1_2.get_loss(pred1_2, correct1, correct2, choice, labels_pl1)
            tf.summary.scalar('loss1_2', loss1_2)
            tf.summary.scalar('otherloss', otherloss)

            correct1_2 = tf.equal(tf.argmax(pred1_2, 2), tf.to_int64(labels_pl1))
            accuracy1_2 = tf.reduce_sum(tf.cast(correct1_2, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy1_2', accuracy1_2)

            loss = 0.4 * loss1 + 0.4 * loss2 + 0.4 * loss1_2 + 0.1 * otherloss
            tf.summary.scalar('loss', loss)
            
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        #saver_1.restore(sess, 'log_2019_8_14/best_model_epoch_390.ckpt')
        #saver_2.restore(sess, 'log_2019_8_15/best_model_epoch_300.ckpt')
        saver.restore(sess, 'log_8192_16384/best_model_epoch_191.ckpt')

        ops = {'pointclouds_pl1': pointclouds_pl1,
               'labels_pl1': labels_pl1,
               'smpws_pl1': smpws_pl1,
               'is_training_pl1': is_training_pl1,
               'pred1': pred1,
               'pointclouds_pl2': pointclouds_pl2,
               'labels_pl2': labels_pl2,
               'smpws_pl2': smpws_pl2,
               'is_training_pl2': is_training_pl2,
               'pred2': pred2,
               'choice':choice,
               'is_training_pl1_2':is_training_pl1_2,
               'pred1_2': pred1_2,
               'loss': loss}

        eval_whole_scene_one_epoch(sess, ops)

def get_batch_3(idx):
    batch_data2, batch_label2, batch_smpw2 = TEST_DATASET_WHOLE_SCENE[idx]
    current_data_item = []
    current_label_item = []
    current_smpw_item = []
    current_choice = []
    for i in range(batch_data2.shape[0]):
        data_item = batch_data2[i,:,:]
        label_item = batch_label2[i,:]
        smpw_item = batch_smpw2[i,:]
        choice = np.random.choice(data_item.shape[0], 8192, replace=False)
        data_item = data_item[choice,:]
        label_item = label_item[choice]
        smpw_item = smpw_item[choice]
        current_data_item.append(data_item)
        current_label_item.append(label_item)
        current_smpw_item.append(smpw_item)
        current_choice.append(choice)
    current_data_item=np.array(current_data_item)
    current_label_item=np.array(current_label_item)
    current_smpw_item=np.array(current_smpw_item)
    current_choice=np.array(current_choice)
    return current_data_item,current_label_item,current_smpw_item,batch_data2,batch_label2,batch_smpw2,current_choice

def eval_whole_scene_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    
    for batch_idx in range(num_batches):
        batch_data1, batch_label1, batch_smpw1, batch_data2, batch_label2, batch_smpw2, choice = get_batch_3(batch_idx)
        result_right_path = "./result/"+str(batch_idx)+"_cv_right.obj"
        result_pre_path = "./result/"+str(batch_idx)+"_cv_pre.obj"
        result_diff_path = "./result/"+str(batch_idx)+"_cv_diff.obj"
        for i in range(batch_data1.shape[0]):
            feed_dict = {ops['pointclouds_pl1']: batch_data1[i:i+1,...],
                         ops['labels_pl1']: batch_label1[i:i+1,...],
                         ops['smpws_pl1']:batch_smpw1[i:i+1,...],
                         ops['is_training_pl1']: is_training,
                         ops['pointclouds_pl2']: batch_data2[i:i+1,...],
                         ops['labels_pl2']: batch_label2[i:i+1,...],
                         ops['smpws_pl2']:batch_smpw2[i:i+1,...],
                         ops['is_training_pl2']: is_training,
                         ops['choice']: choice[i:i+1,...],
                         ops['is_training_pl1_2']:is_training}
            
            pred_val = sess.run([ops['pred1_2']], feed_dict=feed_dict)
            pred_val=np.squeeze(pred_val)
            pred_val = np.argmax(pred_val,axis=1) # BxN
            output_color_point_cloud(batch_data1[i,...],pred_val,result_pre_path)
            output_color_point_cloud(batch_data1[i,...],batch_label1[i,...],result_right_path)
            output_color_point_cloud_red_blue(batch_data1[i,...],pred_val==batch_label1[i,...],result_diff_path)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()

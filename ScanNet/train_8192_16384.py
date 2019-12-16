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
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=2010, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL1 = importlib.import_module(FLAGS.model+"_1")
MODEL2 = importlib.import_module(FLAGS.model+"_2") # import network module
MODEL1_2 = importlib.import_module(FLAGS.model+"_1_2")

MODEL_FILE = os.path.join(os.path.join(ROOT_DIR, 'models'), FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_8192_16384.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 21

# Shapenet official train/test split
DATA_PATH = os.path.join(DATA_ROOT_DIR,'data')
DATA_PATH = os.path.join(DATA_PATH,'scannet')
TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT*2, split='train')
TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT*2, split='test')
TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT*2, split='test')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #2*2 8192
            pointclouds_pl1, labels_pl1, smpws_pl1  = MODEL1.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl1 = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred1, end_points1 = MODEL1.get_model(pointclouds_pl1, is_training_pl1, NUM_CLASSES, bn_decay=bn_decay)
            loss1 = MODEL1.get_loss(pred1, labels_pl1)
            tf.summary.scalar('loss1', loss1)

            correct1 = tf.equal(tf.argmax(pred1, 2), tf.to_int64(labels_pl1))
            accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy1', accuracy1)

            #2*2 16384
            pointclouds_pl2, labels_pl2, smpws_pl2  = MODEL2.placeholder_inputs(BATCH_SIZE, NUM_POINT*2)
            is_training_pl2 = tf.placeholder(tf.bool, shape=())
            

            # Get model and loss 
            pred2, end_points2 = MODEL2.get_model(pointclouds_pl2, is_training_pl2, NUM_CLASSES, bn_decay=bn_decay)
            loss2 = MODEL2.get_loss(pred2, labels_pl2)
            tf.summary.scalar('loss2', loss2)

            correct2 = tf.equal(tf.argmax(pred2, 2), tf.to_int64(labels_pl2))
            accuracy2 = tf.reduce_sum(tf.cast(correct2, tf.float32)) / float(BATCH_SIZE*NUM_POINT*2)
            tf.summary.scalar('accuracy2', accuracy2)

            #  1  Combination  2
            choice  = MODEL1_2.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl1_2 = tf.placeholder(tf.bool, shape=())
            

            # Get model and loss 
            pred1_2, end_points1_2 = MODEL1_2.get_model(end_points1, end_points2, choice, is_training_pl1_2, NUM_CLASSES, bn_decay=bn_decay)
            loss1_2, otherloss = MODEL1_2.get_loss(pred1_2, correct1, correct2, choice, labels_pl1)
            tf.summary.scalar('loss1_2', loss1_2)
            tf.summary.scalar('otherloss', otherloss)

            correct1_2 = tf.equal(tf.argmax(pred1_2, 2), tf.to_int64(labels_pl1))
            accuracy1_2 = tf.reduce_sum(tf.cast(correct1_2, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy1_2', accuracy1_2)

            loss = 0.4 * loss1 + 0.4 * loss2 + 0.4 * loss1_2 + 0.6 * otherloss
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            model1_name = ['layer1_1','layer2_1','layer3_1','layer4_1','fa_layer1_1','fa_layer2_1','fa_layer3_1','fa_layer4_1','fc1_1','dp1_1','fc2_1']
            model1_var = []
            for var in tf.global_variables():
                for var_1 in model1_name:
                    if(var_1 in var.name):
                        model1_var.append(var)
            # Add ops to save and restore all the variables.
            saver_1 = tf.train.Saver(model1_var)
            model2_name = ['layer1_2','layer2_2','layer3_2','layer4_2','fa_layer1_2','fa_layer2_2','fa_layer3_2','fa_layer4_2','fc1_2','dp1_2','fc2_2']
            model2_var = []
            for var in tf.global_variables():
                for var_1 in model2_name:
                    if(var_1 in var.name):
                        model2_var.append(var)
            # Add ops to save and restore all the variables.
            saver_2 = tf.train.Saver(model2_var)
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        saver_1.restore(sess, 'log_8192/best_model_epoch_390.ckpt')
        saver_2.restore(sess, 'log_16384/best_model_epoch_300.ckpt')

        #ckpt = tf.train.get_checkpoint_state("log")
        #if ckpt and ckpt.model_checkpoint_path:
        #    saver.restore(sess,ckpt.model_checkpoint_path)
        #sess.run(init, {is_training_pl: True})

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
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            acc = eval_whole_scene_one_epoch(sess, ops, test_writer)
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data1 = np.zeros((bsize, NUM_POINT, 3))
    batch_label1 = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw1 = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    batch_data2 = np.zeros((bsize, NUM_POINT*2, 3))
    batch_label2 = np.zeros((bsize, NUM_POINT*2), dtype=np.int32)
    batch_smpw2 = np.zeros((bsize, NUM_POINT*2), dtype=np.float32)
    batch_choice = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data2[i,...] = ps
        batch_label2[i,:] = seg
        batch_smpw2[i,:] = smpw

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data2[i,drop_idx,:] = batch_data2[i,0,:]
        batch_label2[i,drop_idx] = batch_label2[i,0]
        batch_smpw2[i,drop_idx] *= 0

        choice = np.random.choice(ps.shape[0], 8192, replace=False)
        ps = ps[choice,:]
        seg = seg[choice]
        smpw = smpw[choice]
        batch_data1[i,...] = ps
        batch_label1[i,:] = seg
        batch_smpw1[i,:] = smpw
        batch_choice[i,:]=choice

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data1[i,drop_idx,:] = batch_data1[i,0,:]
        batch_label1[i,drop_idx] = batch_label1[i,0]
        batch_choice[i,drop_idx] = batch_choice[i,0]
        batch_smpw1[i,drop_idx] *= 0

    return batch_data1, batch_label1, batch_smpw1, batch_data2, batch_label2, batch_smpw2, batch_choice

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

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    total_correct1 = 0
    total_correct2 = 0
    total_correct12 = 0
    total_seen1 = 0
    total_seen2 = 0
    loss_sum = 0
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data1, batch_label1, batch_smpw1, batch_data2, batch_label2, batch_smpw2, choice  = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        aug_data1 = provider.rotate_point_cloud_z(batch_data1)
        aug_data2 = provider.rotate_point_cloud_z(batch_data2)
        feed_dict = {ops['pointclouds_pl1']: aug_data1,
                     ops['labels_pl1']: batch_label1,
                     ops['smpws_pl1']:batch_smpw1,
                     ops['is_training_pl1']: True,
                     ops['pointclouds_pl2']: aug_data2,
                     ops['labels_pl2']: batch_label2,
                     ops['smpws_pl2']:batch_smpw2,
                     ops['is_training_pl2']: True,
                     ops['choice']: choice,
                     ops['is_training_pl1_2']:is_training}
        summary, step, _, loss_val, pred_val1, pred_val2, pred_val12 = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred1'], ops['pred2'], ops['pred1_2']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val1 = np.argmax(pred_val1, 2)
        correct1 = np.sum(pred_val1 == batch_label1)
        total_correct1 += correct1

        pred_val2 = np.argmax(pred_val2, 2)
        correct2 = np.sum(pred_val2 == batch_label2)
        total_correct2 += correct2

        pred_val12 = np.argmax(pred_val12, 2)
        correct12 = np.sum(pred_val12 == batch_label1)
        total_correct12 += correct12

        total_seen1 += (BATCH_SIZE*NUM_POINT)
        total_seen2 += (BATCH_SIZE*NUM_POINT*2)
        loss_sum += loss_val
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('model1 accuracy: %f' % (total_correct1 / float(total_seen1)))
            log_string('model2 accuracy: %f' % (total_correct2 / float(total_seen2)))
            log_string('model12 accuracy: %f' % (total_correct12 / float(total_seen1)))
            total_correct1 = 0
            total_correct2 = 0
            total_correct12 = 0
            total_seen1 = 0
            total_seen2 = 0
            loss_sum = 0

# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET_WHOLE_SCENE))
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION WHOLE SCENE----'%(EPOCH_CNT))

    labelweights = np.zeros(21)
    labelweights_vox = np.zeros(21)
    is_continue_batch = False
    
    extra_batch_data1 = np.zeros((0,NUM_POINT,3))
    extra_batch_label1 = np.zeros((0,NUM_POINT))
    extra_batch_smpw1 = np.zeros((0,NUM_POINT))
    extra_batch_data2 = np.zeros((0,NUM_POINT*2,3))
    extra_batch_label2 = np.zeros((0,NUM_POINT*2))
    extra_batch_smpw2 = np.zeros((0,NUM_POINT*2))
    extra_batch_choice = np.zeros((0,NUM_POINT))
    idx = 0
    for batch_idx in range(num_batches):
        if extra_batch_data1.shape[0]>=BATCH_SIZE:
            batch_data1 = extra_batch_data1
            batch_label1 = extra_batch_label1
            batch_smpw1 = extra_batch_smpw1
            batch_data2 = extra_batch_data2
            batch_label2 = extra_batch_label2
            batch_smpw2 = extra_batch_smpw2
            choice = extra_batch_choice
        elif not is_continue_batch:
            batch_data1, batch_label1, batch_smpw1, batch_data2, batch_label2, batch_smpw2, choice = get_batch_3(idx)
            idx = idx + 1
            batch_data1 = np.concatenate((batch_data1,extra_batch_data1),axis=0)
            batch_label1 = np.concatenate((batch_label1,extra_batch_label1),axis=0)
            batch_smpw1 = np.concatenate((batch_smpw1,extra_batch_smpw1),axis=0)
            batch_data2 = np.concatenate((batch_data2,extra_batch_data2),axis=0)
            batch_label2 = np.concatenate((batch_label2,extra_batch_label2),axis=0)
            batch_smpw2 = np.concatenate((batch_smpw2,extra_batch_smpw2),axis=0)
            choice = np.concatenate((choice,extra_batch_choice),axis=0)
        else:
            batch_data_tmp1, batch_label_tmp1, batch_smpw_tmp1, batch_data_tmp2, batch_label_tmp2, batch_smpw_tmp2, choice_tmp = get_batch_3(idx)
            idx = idx + 1
            batch_data1 = np.concatenate((batch_data1,batch_data_tmp1),axis=0)
            batch_label1 = np.concatenate((batch_label1,batch_label_tmp1),axis=0)
            batch_smpw1 = np.concatenate((batch_smpw1,batch_smpw_tmp1),axis=0)
            batch_data2 = np.concatenate((batch_data2,batch_data_tmp2),axis=0)
            batch_label2 = np.concatenate((batch_label2,batch_label_tmp2),axis=0)
            batch_smpw2 = np.concatenate((batch_smpw2,batch_smpw_tmp2),axis=0)
            choice = np.concatenate((choice,choice_tmp),axis=0)
        if batch_data1.shape[0]<BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data1.shape[0]==BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data1 = np.zeros((0,NUM_POINT,3))
            extra_batch_label1 = np.zeros((0,NUM_POINT))
            extra_batch_smpw1 = np.zeros((0,NUM_POINT))
            extra_batch_data2 = np.zeros((0,NUM_POINT*2,3))
            extra_batch_label2 = np.zeros((0,NUM_POINT*2))
            extra_batch_smpw2 = np.zeros((0,NUM_POINT*2))
            extra_batch_choice = np.zeros((0,NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data1 = batch_data1[BATCH_SIZE:,:,:]
            extra_batch_label1 = batch_label1[BATCH_SIZE:,:]
            extra_batch_smpw1 = batch_smpw1[BATCH_SIZE:,:]
            extra_batch_data2 = batch_data2[BATCH_SIZE:,:,:]
            extra_batch_label2 = batch_label2[BATCH_SIZE:,:]
            extra_batch_smpw2 = batch_smpw2[BATCH_SIZE:,:]
            extra_batch_choice = choice[BATCH_SIZE:,:]
            batch_data1 = batch_data1[:BATCH_SIZE,:,:]
            batch_label1 = batch_label1[:BATCH_SIZE,:]
            batch_smpw1 = batch_smpw1[:BATCH_SIZE,:]
            batch_data2 = batch_data2[:BATCH_SIZE,:,:]
            batch_label2 = batch_label2[:BATCH_SIZE,:]
            batch_smpw2 = batch_smpw2[:BATCH_SIZE,:]
            choice = choice[:BATCH_SIZE,:]

        feed_dict = {ops['pointclouds_pl1']: batch_data1,
                     ops['labels_pl1']: batch_label1,
                     ops['smpws_pl1']:batch_smpw1,
                     ops['is_training_pl1']: is_training,
                     ops['pointclouds_pl2']: batch_data2,
                     ops['labels_pl2']: batch_label2,
                     ops['smpws_pl2']:batch_smpw2,
                     ops['is_training_pl2']: is_training,
                     ops['choice']: choice,
                     ops['is_training_pl1_2']:is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred1_2']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label1) & (batch_label1>0) & (batch_smpw1>0)) # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label1>0) & (batch_smpw1>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label1,range(22))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label1==l) & (batch_smpw1>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label1==l) & (batch_smpw1>0))

        for b in range(batch_label1.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(batch_data1[b,batch_smpw1[b,:]>0,:], np.concatenate((np.expand_dims(batch_label1[b,batch_smpw1[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw1[b,:]>0],1)),axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
            tmp,_ = np.histogram(uvlabel[:,0],range(22))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    log_string('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval whole scene point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
    log_string('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
    log_string('eval whole scene point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    caliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)
    log_string('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------'
    for l in range(1,NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (l,labelweights_vox[l-1],total_correct_class_vox[l]/float(total_seen_class_vox[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return caliacc


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()

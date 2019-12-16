import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
first_DIR = os.path.dirname(BASE_DIR)
DARA_DIR = os.path.join(first_DIR, 'data')
DARA_DIR = os.path.join(DARA_DIR, 'S3DIS_16384')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='sem_seg_2', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models/'+FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_8192.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

ALL_FILES = provider.getDataFiles(os.path.join(DARA_DIR, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))
room_filelist = [line.rstrip() for line in open(os.path.join(DARA_DIR,'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(os.path.join(DARA_DIR, h5_filename))
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)

test_area = 'Area_'+str(FLAGS.test_area)
train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs,:,:]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,:,:]
test_label = label_batches[test_idxs]
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)




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
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
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
            pointclouds_pl, labels_pl  = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_acc=-1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer)
            if epoch%5==0:
                acc = eval_one_epoch(sess, ops, test_writer)
            if acc > best_acc:
                best_acc = acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data, train_label)
    print(current_data.shape,current_label.shape)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        current_data_item = []
        current_choice = []
        current_label_item = []
        for i in range(BATCH_SIZE):
            data_item = current_data[start_idx+i,:,:]
            label_item = current_label[start_idx+i,:]
            choice_1 = np.random.choice(8192, 4096, replace=False)
            choice_2 = np.random.choice(8192, 4096, replace=False)
            choice_2 = choice_2+8192
            choice = np.concatenate((choice_1,choice_2),axis=0)
            data_item = data_item[choice,:]
            label_item = label_item[choice]
            current_data_item.append(data_item)
            current_label_item.append(label_item)
            current_choice.append(choice)
        current_data_item=np.array(current_data_item)
        current_label_item=np.array(current_label_item)

        feed_dict = {ops['pointclouds_pl']: current_data_item,
                     ops['labels_pl']: current_label_item,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label_item)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data
    current_label = np.squeeze(test_label)
    print(current_data.shape,current_label.shape)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        current_data_item = []
        current_choice = []
        current_label_item = []
        for i in range(BATCH_SIZE):
            data_item = current_data[start_idx+i,:,:]
            label_item = current_label[start_idx+i,:]
            choice_1 = np.random.choice(8192, 4096, replace=False)
            choice_2 = np.random.choice(8192, 4096, replace=False)
            choice_2 = choice_2+8192
            choice = np.concatenate((choice_1,choice_2),axis=0)
            data_item = data_item[choice,:]
            label_item = label_item[choice]
            current_data_item.append(data_item)
            current_label_item.append(label_item)
            current_choice.append(choice)
        current_data_item=np.array(current_data_item)
        current_label_item=np.array(current_label_item)

        feed_dict = {ops['pointclouds_pl']: current_data_item,
                     ops['labels_pl']: current_label_item,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label_item)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(BATCH_SIZE):
            for j in range(NUM_POINT):
                l = current_label_item[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    result_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    log_string('eval avg class acc: %f' % result_acc)
    return result_acc
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()

import argparse
import os
import sys
import tensorflow as tf
import importlib
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
first_DIR = os.path.dirname(BASE_DIR)
DARA_DIR = os.path.join(first_DIR, 'data')
DARA_DIR = os.path.join(DARA_DIR, 'S3DIS_16384')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
import indoor3d_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='sem_seg_1', help='Model name [default: model]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--model_path',default='./log_4096_rgb/best_model_epoch_610.ckpt', help='model checkpoint file path')
parser.add_argument('--dump_dir', default='./result_4096', help='dump folder path')
parser.add_argument('--room_data_filelist', default='meta/area6_data_label.txt', help='TXT filename, filelist, each line is a test room data label file.')
FLAGS = parser.parse_args()

MODEL = importlib.import_module(FLAGS.model)
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
ROOM_PATH_LIST = [os.path.join(DARA_DIR,line.rstrip()) for line in open(FLAGS.room_data_filelist)]

NUM_CLASSES = 13

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred,end_points = MODEL.get_model(pointclouds_pl, is_training_pl,NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    for room_path in ROOM_PATH_LIST:
        out_data_label_filename = os.path.basename(room_path)[:-4] + '_pred.obj'
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        out_gt_label_filename = os.path.basename(room_path)[:-4] + '_gt.obj'
        out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)
        out_diff_label_filename = os.path.basename(room_path)[:-4] + '_diff.obj'
        out_diff_label_filename = os.path.join(DUMP_DIR, out_diff_label_filename)
        log_string(out_data_label_filename)
        a, b, c, d = eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename,out_diff_label_filename)
        total_correct += a
        total_seen += b
        for i in range(NUM_CLASSES):
            total_seen_class[i]+=d[i]
            total_correct_class[i]+=c[i]
        print(total_seen_class)
        print(total_correct_class)
    for l in range(NUM_CLASSES):
        log_string('class %d, acc: %f;' % (l,total_correct_class[l]/float(total_seen_class[l])))
    log_string('all room eval accuracy: %f'% (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, room_path, out_data_label_filename, out_gt_label_filename, out_diff_label_filename):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout_data_label = open(out_data_label_filename, 'w')
    fout_gt_label = open(out_gt_label_filename, 'w')
    fout_diff_label = open(out_diff_label_filename, 'w')
    
    current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT)
    current_data = current_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(current_label)
    # Get room dimension..
    data_label = np.load(room_path)
    data = data_label[:,0:6]
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)

    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
        pred_label = np.argmax(pred_val, 2) # BxN
        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data[start_idx+b, :, :]
            l = current_label[start_idx+b,:]
            pts[:,6] *= max_room_x
            pts[:,7] *= max_room_y
            pts[:,8] *= max_room_z
            pts[:,3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(NUM_POINT):
                color = indoor3d_util.g_label2color[pred[i]]
                if(pred[i]==current_label[start_idx+b, i]):
                	color_diff=[9,200,248]
                else:
                	color_diff=[1,0,0]
                color_gt = indoor3d_util.g_label2color[current_label[start_idx+b, i]]
                fout_data_label.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color[0], color[1], color[2]))
                fout_gt_label.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_gt[0], color_gt[1], color_gt[2]))
                fout_diff_label.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_diff[0], color_diff[1], color_diff[2]))
        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    #fout_data_label.close()
    #fout_gt_label.close()
    return total_correct, total_seen, total_correct_class, total_seen_class


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()

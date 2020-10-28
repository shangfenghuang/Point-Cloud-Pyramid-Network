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
parser.add_argument('--model', default='sem_seg', help='Model name [default: model]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=8192, help='Point number [default: 4096]')
parser.add_argument('--model_path',default='./log_4096_8192_rgb/best_model_epoch_172.ckpt', help='model checkpoint file path')
parser.add_argument('--dump_dir', default='./result_4096_8192', help='dump folder path')
parser.add_argument('--room_data_filelist', default='meta/area6_data_label.txt', help='TXT filename, filelist, each line is a test room data label file.')
FLAGS = parser.parse_args()

MODEL1 = importlib.import_module(FLAGS.model+"_1")
MODEL2 = importlib.import_module(FLAGS.model+"_2")
MODEL1_2 = importlib.import_module(FLAGS.model+"_1_2")

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
        #2*2 4096
        pointclouds_pl1, labels_pl1  = MODEL1.placeholder_inputs(BATCH_SIZE, NUM_POINT/2)
        is_training_pl1 = tf.placeholder(tf.bool, shape=())
        pred1, end_points1 = MODEL1.get_model(pointclouds_pl1, is_training_pl1, NUM_CLASSES)
        loss1 = MODEL1.get_loss(pred1, labels_pl1)
        correct1 = tf.equal(tf.argmax(pred1, 2), tf.to_int64(labels_pl1))
        accuracy1 = tf.reduce_sum(tf.cast(correct1, tf.float32)) / float(BATCH_SIZE*NUM_POINT/2)
        
        #2*2 8192
        pointclouds_pl2, labels_pl2  = MODEL2.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl2 = tf.placeholder(tf.bool, shape=())
        pred2, end_points2 = MODEL2.get_model(pointclouds_pl2, is_training_pl2, NUM_CLASSES)
        loss2 = MODEL2.get_loss(pred2, labels_pl2)
        correct2 = tf.equal(tf.argmax(pred2, 2), tf.to_int64(labels_pl2))
        accuracy2 = tf.reduce_sum(tf.cast(correct2, tf.float32)) / float(BATCH_SIZE*NUM_POINT)

        #  1  Combination  2
        choice1  = MODEL1_2.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl1_2 = tf.placeholder(tf.bool, shape=())
        pred1_2, end_points1_2 = MODEL1_2.get_model(end_points1, end_points2, choice1, is_training_pl1_2, NUM_CLASSES)
        pred_softmax = tf.nn.softmax(pred1_2)
        loss1_2, otherloss = MODEL1_2.get_loss(pred1_2, correct1, correct2, choice1, labels_pl1)
        correct1_2 = tf.equal(tf.argmax(pred1_2, 2), tf.to_int64(labels_pl1))
        accuracy1_2 = tf.reduce_sum(tf.cast(correct1_2, tf.float32)) / float(BATCH_SIZE*NUM_POINT/2)
        
        loss = 0.1 * loss1 + 0.2 * loss2 + 0.4 * loss1_2 + 0.3 * otherloss
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

    ops = { 'pointclouds_pl1': pointclouds_pl1,
            'labels_pl1': labels_pl1,
            'is_training_pl1': is_training_pl1,
            'pred1': pred1,
            'pointclouds_pl2': pointclouds_pl2,
            'labels_pl2': labels_pl2,
            'is_training_pl2': is_training_pl2,
            'pred2': pred2,
            'choice1':choice1,
            'is_training_pl1_2':is_training_pl1_2,
            'pred1_2': pred1_2,
            'loss': loss,
            'pred_softmax': pred_softmax}
    
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
    
    current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT*2)
    current_data = current_data
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

        current_data_item1 = []
        current_choice1 = []
        current_label_item1 = []
        current_data_item2 = []
        current_choice2 = []
        current_label_item2 = []
        for i in range(BATCH_SIZE):
            data_item = current_data[start_idx+i,:,:]
            label_item = current_label[start_idx+i,:]
            choice_1 = np.random.choice(8192, 4096, replace=False)
            choice_2 = np.random.choice(8192, 4096, replace=False)
            choice_2 = choice_2+8192
            choice = np.concatenate((choice_1,choice_2),axis=0)
            data_item = data_item[choice,:]
            label_item = label_item[choice]
            current_data_item2.append(data_item)
            current_label_item2.append(label_item)
            current_choice2.append(choice)
            choice_3 = np.random.choice(8192, 4096, replace=False)
            data_item = data_item[choice_3,:]
            label_item = label_item[choice_3]
            current_data_item1.append(data_item)
            current_label_item1.append(label_item)
            current_choice1.append(choice_3)
        current_data_item1=np.array(current_data_item1)#b * 4096 * 3
        current_label_item1=np.array(current_label_item1)#b * 4096
        current_choice1=np.array(current_choice1)# b * 4096
        current_data_item2=np.array(current_data_item2)#b * 8192 * 3
        current_label_item2=np.array(current_label_item2)#b * 8192
        current_choice2=np.array(current_choice2)# b * 8192

        feed_dict = {ops['pointclouds_pl1']: current_data_item1[:,:,:],
                     ops['labels_pl1']: current_label_item1,
                     ops['is_training_pl1']: False,
                     ops['pointclouds_pl2']: current_data_item2[:,:,:],
                     ops['labels_pl2']: current_label_item2,
                     ops['is_training_pl2']: False,
                     ops['choice1']:current_choice1,
                     ops['is_training_pl1_2']:False}
        
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
        pred_label = np.argmax(pred_val, 2) # BxN
        # Save prediction labels to OBJ file
        for b in range(BATCH_SIZE):
            pts = current_data_item1[b, :, :]
            l = current_label_item1[b,:]
            pts[:,6] *= max_room_x
            pts[:,7] *= max_room_y
            pts[:,8] *= max_room_z
            pts[:,3:6] *= 255.0
            pred = pred_label[b, :]
            for i in range(int(NUM_POINT/2)):
                color = indoor3d_util.g_label2color[pred[i]]
                if(pred[i]==current_label_item1[b, i]):
                	color_diff=[9,200,248]
                else:
                	color_diff=[1,0,0]
                color_gt = indoor3d_util.g_label2color[current_label_item1[b, i]]
                fout_data_label.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color[0], color[1], color[2]))
                fout_gt_label.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_gt[0], color_gt[1], color_gt[2]))
                fout_diff_label.write('v %f %f %f %d %d %d\n' % (pts[i,6], pts[i,7], pts[i,8], color_diff[0], color_diff[1], color_diff[2]))
        correct = np.sum(pred_label == current_label_item1[:,:])
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT/2)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(BATCH_SIZE):
            for j in range(int(NUM_POINT/2)):
                l = current_label_item1[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT/2)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    #fout_data_label.close()
    #fout_gt_label.close()
    return total_correct, total_seen, total_correct_class, total_seen_class


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import DSA_module, pointnet_fp_module

def placeholder_inputs(batch_size, num_point):
    choice1 = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return choice1

def get_model(end_points1, end_points2, choice1, is_training, num_class, bn_decay=None):
    item = end_points2['feats'][0]
    choice2 = choice1[0]
    end_points2_item = tf.gather(item,choice2,axis=0)
    end_points2_item = tf.expand_dims(end_points2_item,0)
    for i in range(1,choice1.shape[0]):
        item = end_points2['feats'][i]
        choice2 = choice1[i]
        item = tf.gather(item,choice2,axis=0)
        item = tf.expand_dims(item,0)
        end_points2_item = tf.concat([end_points2_item,item],0)
    end_points2 = end_points2_item
    end_points = tf.concat([end_points1['feats'],end_points2],2)
    net = tf_util.conv1d(end_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fc112', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc212', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc312', bn_decay=bn_decay)
    end_points = {}
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp112')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc412')
    return net, end_points


def get_loss(pred1_2, correct1, correct2, choice1, label):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred1_2)
    
    item = correct2[0]
    choice2 = choice1[0]
    correct2_item = tf.gather(item,choice2,axis=0)
    correct2_item = tf.expand_dims(correct2_item,0)
    for i in range(1,choice1.shape[0]):
        item = correct2[i]
        choice2 = choice1[i]
        item = tf.gather(item,choice2,axis=0)
        item = tf.expand_dims(item,0)
        correct2_item = tf.concat([correct2_item,item],0)
    correct2 = correct2_item
    correct = (2 - tf.cast(correct1, tf.float32) - tf.cast(correct2, tf.float32)) * 0.5
    other_loss = tf.reduce_mean(correct)
    return classify_loss, other_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)

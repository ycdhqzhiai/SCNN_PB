#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math
import numpy as np
import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import sys
sys.path.append('D:\workspace\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model')
from lanenet_model import lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
def color_options(x):
    return {
        1: (0, 255, 0), # green color
        2: (255, 0, 0), # blue
        3: (0, 0, 255), # red
        4: (0, 0, 0)
    }[x]

def process(img):
    img_resized = cv2.resize(img, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),interpolation=cv2.INTER_CUBIC)     
    sub_img = img_resized.astype('float32') - VGG_MEAN 
    image = np.expand_dims(sub_img.astype('float32'), 0)
    return image

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir',default='/home/pch/test/SCNN/testlinux.txt')
    parser.add_argument('--weights_path', type=str, help='The model weights path', default='/home/pch/test/SCNN/culane_lanenet_vgg_2018-12-17-19-36-42.ckpt-90000')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=8)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def test_lanenet(image_path, weights_path, use_gpu, image_list, batch_size):
    """

    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 288, 800, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.test_inference(input_tensor, phase_tensor, 'lanenet_loss')

    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)
    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=weights_path)
        img = np.zeros((1,288,800,3))
        instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],feed_dict={input_tensor: img})
        print(instance_seg_image, existence_output)

        saver.save(sess, './models/tmp/tmp.ckpt')
    sess.close()
    
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    img_name = []
    with open(str(args.image_path), 'r') as g:
        for line in g.readlines():
            img_name.append(line.strip())

    test_lanenet(args.image_path, args.weights_path, args.use_gpu, img_name, args.batch_size)

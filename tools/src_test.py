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
import math
import numpy as np
import tensorflow as tf
import glog as log
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


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=8)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()



def test_lanenet(image_path, weights_path, use_gpu, image_list, batch_size, save_dir):

    """
    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    src_image = cv2.imread('./00330.jpg') 
    
    #print(src_image.shape[0], src_image.shape[1])
    #src_image = cv2.resize(src_image, (1640,590), interpolation=cv2.INTER_AREA) 
    w ,h= src_image.shape[:2] 
    test_dataset = lanenet_data_processor_test.DataSet(image_path, batch_size)
    input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')

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
        for i in range(math.ceil(len(image_list) / batch_size)):
            #print(i)
            paths = test_dataset.next_batch()
            instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: paths})
            for cnt, image_name in enumerate(paths):
                #print(image_name)
                parent_path = os.path.dirname(image_name)
                directory = os.path.join(save_dir, 'vgg_SCNN_DULR_w9', parent_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_exist = open(os.path.join(directory, os.path.basename(image_name)[:-3] + 'exist.txt'), 'w')
                for cnt_img in range(4):
                    coordinate_tmp = np.zeros((1,30))
                    #print(coordinate_tmp)

                    img = (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int)
                    #print(w, h)
                    for i in range(30):
                        #print(i)
                        lineId = math.ceil(288- i*20/w*288) - 1 
                        #print(lineId)
                        #print(img.shape)
                        img_line = img[lineId]                        
                        #print(type(img_line))a = list(a)
                        value = np.max(img_line)
                        id = np.where(img_line==value)
                        #print(id[0])
                        #Wprint()
                        #print(value, id[0][0])
                        if(value / 255 > 0.3):
                            coordinate_tmp[0][i] = id[0][0]
                    
                    if np.sum(coordinate_tmp>0) < 2:
                        coordinate_tmp = np.zeros((1,30))
                    #print(coordinate_tmp)
                    #print(np.sum(coordinate_tmp>0))
                    for i in range(30):
                        if coordinate_tmp[0][i]>0:
                            cv2.circle(src_image, (int(coordinate_tmp[0][i]*h/800), int(w-i*20)), 6, (0,0,255),-1) 
                        #line = score(lineId,:)
                    #[value, id] = max(line)
                    #if double(value)/255 > thr
                    #    coordinate(i) = id
                    #end
                    #end                    
                    cv2.imwrite(os.path.join(directory, os.path.basename(image_name)[:-4] + '_' + str(cnt_img + 1) + '_avg.png'),
                            (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int))
                    if existence_output[cnt, cnt_img] > 0.5:
                        print('\n\n\n')
                        print(existence_output[cnt, cnt_img])
                        print('\n\n\n')
                        file_exist.write('1 ')
                    else:
                        file_exist.write('0 ')
                file_exist.close()
            cv2.imshow('img', src_image)
            cv2.waitKey(0)
    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    #save_dir = os.path.join(args.image_path, 'predicts')
    #if args.save_dir is not None:
    #    save_dir = args.save_dir

    img_name = []
    with open(str(args.image_path), 'r') as g:
        for line in g.readlines():
            img_name.append(line.strip())

    test_lanenet(args.image_path, args.weights_path, args.use_gpu, img_name, args.batch_size, args.save_dir)
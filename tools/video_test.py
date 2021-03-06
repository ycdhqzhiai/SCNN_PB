import os
import os.path as ops
import argparse
import math
import tensorflow as tf
import glog as log
import cv2
import numpy as np
import sys
sys.path.append('D:\workspace\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model')
from lanenet_model import lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test


CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


class DataProcess(object):
    """
    实现数据集类
    """

    def __init__(self, img):
        """

        :param dataset_info_file:
        """
        self._img = img

    @staticmethod
    def process_img(img):
        print ('img_path', img)
        #img_raw = tf.read_file(img_path)
        #img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
        img_resized = tf.image.resize_images(img, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH],
                                             method=tf.image.ResizeMethod.BICUBIC)
        img_casted = tf.cast(img_resized, tf.float32)
        print (img_casted)
        return tf.subtract(img_casted, VGG_MEAN)

def color_options(x):
    return {
        1: (0, 255, 0), # green color
        2: (255, 0, 0), # blue
        3: (0, 0, 255), # red
        4: (0, 0, 0)
    }[x]

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

def process(img):
    img_resized = cv2.resize(img, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),interpolation=cv2.INTER_CUBIC)     
    print ('\n\n\n')
    print (img_resized.shape)
    print ('\n\n\n')
    sub_img = img_resized.astype('float32') - VGG_MEAN 
    image = np.expand_dims(sub_img.astype('float32'), 0)
    return image

def test_lanenet(image_path, weights_path, use_gpu, image_list, batch_size):

    """
    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    print('image_path:',image_path)
    #frame = cv2.imread('obj.jpg')  
    cap = cv2.VideoCapture('harder_challenge_video.mp4') 

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 288, 800, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, phase_tensor=phase_tensor, name='lanenet_loss')
    #binary_seg_ret, instance_seg_ret = net.test_inference(input_tensor=input_tensor, phase_tensor, 'lanenet_loss')
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
        while(True):
            ret, frame = cap.read()
            image = process(frame)
            instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: image})
            for cnt_img in range(4):
                print (existence_output[0, cnt_img])
                if existence_output[0, cnt_img] > 0.5:
                    obj_mask = (instance_seg_image[0, :, :, cnt_img + 1]* 255).astype(int)
                    h_scale = frame.shape[0] / obj_mask.shape[0]
                    w_scale = frame.shape[1] / obj_mask.shape[1]
                                
                    coordinate_tmp = np.zeros((1,30))
                    img = (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int)
                    #print(w, h)
                    for i in range(30):
                        lineId = math.ceil(288- i*20/w*288) - 1 
                        img_line = img[lineId]                        
                        value = np.max(img_line)
                        id = np.where(img_line==value)
                        if(value / 255 > 0.3):
                            coordinate_tmp[0][i] = id[0][0]                    
                    if np.sum(coordinate_tmp>0) < 2:
                        coordinate_tmp = np.zeros((1,30))
                    for i in range(30):
                        if coordinate_tmp[0][i]>0:
                            cv2.circle(src_image, (int(coordinate_tmp[0][i]*h/800), int(w-i*20)), 6, (0,0,255),-1)        
            #cv2.imwrite('obj.jpg', frame)
            cv2.imshow('obj',frame)
            cv2.waitKey(1)
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

    test_lanenet(args.image_path, args.weights_path, args.use_gpu, img_name, args.batch_size)
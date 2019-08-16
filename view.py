import os
import cv2
import numpy as np
import tensorflow as tf
import math

VGG_MEAN = [103.939, 116.779, 123.68]
def process(img):
    img_resized = cv2.resize(img, (800, 288),interpolation=cv2.INTER_CUBIC)     
    sub_img = img_resized.astype('float32') - VGG_MEAN 
    image = np.expand_dims(sub_img.astype('float32'), 0)
    #image = np.vstack(image)
    return image

def test_inference(decode_logits, existence_logit):
    binary_seg_ret = tf.nn.softmax(logits=decode_logits)
    prob_list = []
    kernel = tf.get_variable('kernel', [9, 9, 1, 1], initializer=tf.constant_initializer(1.0 / 81),
                                trainable=False)

    #with tf.variable_scope("convs_smooth"):
    prob_smooth = tf.nn.conv2d(tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, 0], axis=3), tf.float32),
        kernel, [1, 1, 1, 1], 'SAME')
    prob_list.append(prob_smooth)

    for cnt in range(1, binary_seg_ret.get_shape().as_list()[3]):
        #with tf.variable_scope("convs_smooth", reuse=True):
        prob_smooth = tf.nn.conv2d(
            tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, cnt], axis=3), tf.float32), kernel, [1, 1, 1, 1],
            'SAME')
        prob_list.append(prob_smooth)
    processed_prob = tf.stack(prob_list, axis=4)
    processed_prob = tf.squeeze(processed_prob, axis=3)
    binary_seg_ret = processed_prob

    # Predict lane existence:
    #existence_logit = inference_ret['existence_output']
    existence_output = tf.nn.sigmoid(existence_logit)

    return binary_seg_ret, existence_output


src_image = cv2.imread('00330.jpg')
w ,h= src_image.shape[:2]
image = process(src_image)
pb_path = 'tmp.pb'
#data = np.zeros((8, 288, 800, 3), np.float32)  
#b = tf.placeholder(dtype=tf.float32, shape=[8,288,800,5], name='b')
#c = tf.placeholder(dtype=tf.float32, shape=[8,4], name='c')
#instance_seg, existence = test_inference(b, c)


#for idx in range(8):     
#    data[idx, :, :, :] = image  
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        input_tensor = sess.graph.get_tensor_by_name("input_tensor:0")
        existence_output = sess.graph.get_tensor_by_name('lanenet_loss_1/Squeeze:0')
        binary_seg_ret = sess.graph.get_tensor_by_name("lanenet_loss_1/Sigmoid:0")
        #net_phase_placeholder = tf.get_default_graph().get_tensor_by_name("net_phase:0")

        existence_output, instance_seg_image=sess.run([binary_seg_ret, existence_output], feed_dict={input_tensor: image})
        print(instance_seg_image.shape)
        for cnt_img in range(4):
            coordinate_tmp = np.zeros((1,30))
                    #print(coordinate_tmp)

            img = (instance_seg_image[0, :, :, cnt_img + 1] * 255).astype(int)
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
        cv2.imshow('img', src_image)
        cv2.waitKey(0)  
    
    
    
    
    
    
    
    
    

        #instance_seg, existence = test_inference(binary_seg_ret, existence_output)

        #instance_seg, existence = sess.run()
        #out = tf.nn.sigmoid(out)
        #out_numpy=out.eval(session=sess)        
        #instance_seg_image, existence_output = test_inference(instance_seg_image, existence_output)
        #print(instance[0], existence_output)
        #mask=instance_seg_image.eval(session=sess) 
       
        #mask = instance_seg_image.eval() 
        #existence_output=existence_output.eval(session=sess)   
        #print(mask)  
        # for cnt_img in range(4):
        #     coordinate_tmp = np.zeros((1,30))
        #     #print(coordinate_tmp)
        #     img = (instance_seg_image[0, :, :, cnt_img + 1] * 255).astype(int)
        #     print(w, h)
        #     for i in range(30):
        #         lineId = math.ceil(288- i*20/w*288) - 1 
        #         img_line = img[lineId]                        
        #         value = np.max(img_line)
        #         id = np.where(img_line==value)
        #         if(value / 255 > 0.3):
        #             coordinate_tmp[0][i] = id[0][0]
        #         if np.sum(coordinate_tmp>0) < 2:
        #             coordinate_tmp = np.zeros((1,30))
        #         for i in range(30):
        #             if coordinate_tmp[0][i]>0:
        #                 cv2.circle(src_image, (int(coordinate_tmp[0][i]*h/800), int(w-i*20)), 6, (0,0,255),-1) 
        #     cv2.imshow('img', src_image)
        #     cv2.waitKey(0)


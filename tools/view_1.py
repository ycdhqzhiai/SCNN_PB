from tensorflow.python.framework import graph_util
import tensorflow as tf
import argparse
import os
import sys
from six.moves import xrange  # @UnresolvedImport
import sys
sys.path.append('D:\workspace\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model')
from lanenet_model import lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model metagraph and checkpoint
            print('Model directory: %s' % args.model_dir)
            meta_file = 'tmp.ckpt.meta'
            ckpt_file = 'tmp.ckpt'
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            model_dir_exp = os.path.expanduser(args.model_dir)
            print('model_dir_exp file: %s' % model_dir_exp)
            meta_path = os.path.join(model_dir_exp, meta_file)
            print('meta_path file: %s' % meta_path)
            saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
            tf.get_default_session().run(tf.global_variables_initializer())
            tf.get_default_session().run(tf.local_variables_initializer())
            saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))
            

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            input_graph_def = sess.graph.as_graph_def()
         
            # Freeze the graph def
            output_graph_def = freeze_graph_def(sess, input_graph_def, 'lanenet_loss_1/Squeeze,lanenet_loss_1/Sigmoid')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(args.output_file, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph: %s" % (len(output_graph_def.node), args.output_file))
        
def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # Get the list of important nodes
    #whitelist_names = []
    # for node in input_graph_def.node:
    #     if (node.name.startswith('InceptionResnet') or node.name.startswith('embeddings') or 
    #             node.name.startswith('image_batch') or node.name.startswith('label_batch') or
    #             node.name.startswith('phase_train') or node.name.startswith('Logits')):
    #         whitelist_names.append(node.name)

    # Replace all the variables in the graph with constants of the same values
    #output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
    return output_graph_def
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str, 
        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
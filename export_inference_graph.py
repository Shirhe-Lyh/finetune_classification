#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:46:16 2018

@author: shirhe-lyh
"""

"""Tool to export a model for inference.

Outputs inference graph, asscociated checkpoint files, a frozen inference
graph and a SavedModel (https://tensorflow.github.io/serving_basic.html).

The inference graph contains one of three input nodes depending on the user
specified option.
    * 'image_tensor': Accepts a uint8 4-D tensor of shape [None, None, None, 3]
    * 'encoded_image_string_tensor': Accepts a 1-D string tensor of shape 
        [None] containg encoded PNG or JPEG images.
    * 'tf_example': Accepts a 1-D string tensor of shape [None] containing
        serialized TFExample protos.
        
and the following output nodes returned by the model.postprocess(..):
    * 'classes': Outputs float32 tensors of the form [batch_size] containing
        the classes for the predictions.
        
Example Usage:
---------------
python/python3 export_inference_graph \
    --input_type image_tensor \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
    
The exported output would be in the directory
path/to/exported_model_directory (which is created if it does not exist)
with contents:
    - model.ckpt.data-00000-of-00001
    - model.ckpt.info
    - model.ckpt.meta
    - frozen_inference_graph.pb
    + saved_model (a directory)
"""

import os
import tensorflow as tf

import exporter
import model

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can '
                    "be one of ['image_tensor', 'encoded_image_string_tensor'"
                    ", 'tf_example']")
flags.DEFINE_string('input_shape', None, "If input_type is 'image_tensor', "
                    "this can be explicitly set the shape of this input "
                    "to a fixed size. The dimensions are to be provided as a "
                    "comma-seperated list of integers. A value of -1 can be "
                    "used for unknown dimensions. If not specified, for an "
                    "'image_tensor', the default shape will be partially "
                    "specified as '[None, None, None, 3]'.")
flags.DEFINE_string('trained_checkpoint_prefix', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', None, 'Path to write outputs')
tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    cls_model = model.Model(is_training=False, num_classes=61)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != -1 else None 
            for dim in FLAGS.input_shape.split(',')
        ]
    else:
        input_shape = [None, None, None, 3]
    exporter.export_inference_graph(FLAGS.input_type,
                                    cls_model,
                                    FLAGS.trained_checkpoint_prefix,
                                    FLAGS.output_directory,
                                    input_shape)
    

if __name__ == '__main__':
    tf.app.run()


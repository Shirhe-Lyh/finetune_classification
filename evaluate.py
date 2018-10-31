# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:58:14 2018

@author: shirhe-lyh
"""

import cv2
import json
import os
import tensorflow as tf

import data_provider
import predictor

flags = tf.app.flags

flags.DEFINE_string('frozen_inference_graph_path',
                    './training/frozen_inference_graph_pb/'+
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')
flags.DEFINE_string('images_dir', 
                    '/data2/raycloud/jingxiong_datasets/AIChanllenger/' +
                     'AgriculturalDisease_validationset/images', 
                    'Path to images (directory).')
flags.DEFINE_string('annotation_path', 
                    '/data2/raycloud/jingxiong_datasets/' +
                    'AIChanllenger/AgriculturalDisease_validationset/' +
                    'AgriculturalDisease_validation_annotations.json',
                    'Path to annotation`s .json file.')
flags.DEFINE_string('output_path', './val_result.json', 'Path to output file.')

FLAGS = flags.FLAGS


if __name__ == '__main__':
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
    images_dir = FLAGS.images_dir
    annotation_path = FLAGS.annotation_path
    output_path = FLAGS.output_path
    
    model = predictor.Predictor(frozen_inference_graph_path)
    
    _, annotation_dict = data_provider.provide(annotation_path, images_dir)

    val_results = []
    correct_count = 0
    predicted_count = 0
    num_samples = len(annotation_dict)
    for image_path, label in annotation_dict.items():
        predicted_count += 1
        if predicted_count % 100 == 0:
            print('Predict {}/{}.'.format(predicted_count, num_samples))
        
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        if image is None:
            print('image %s does not exist.' % image_name)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        pred_label = int(model.predict([image])[0])
        if pred_label == label:
            correct_count += 1
        d = {}
        d['image_id'] = image_name
        d['disease_class'] = pred_label
        val_results.append(d)
        
    print('Accuracy: ', correct_count*1.0/num_samples)
        
    #pred_results_json = json.dumps(val_results)
    file = open('./val_result_10000.json', 'w')
    json.dump(val_results, file)
    file.close()

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:43:47 2018

@author: shirhe-lyh


Read a .txt file to provide annotation class labels.
"""

import json
import os


def provide(annotation_path=None, images_dir=None):
    """Return image_paths and class labels.
    
    Args:
        annotation_path: Path to an anotation's .json file.
        images_dir: Path to images directory.
            
    Returns:
        image_files: A list containing the paths of images.
        annotation_dict: A dictionary containing the class labels of each 
            image.
            
    Raises:
        ValueError: If annotation_path does not exist.
    """
    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')
        
    annotation_json = open(annotation_path, 'r')
    annotation_list = json.load(annotation_json)
    image_files = []
    annotation_dict = {}
    for d in annotation_list:
        image_name = d.get('image_id')
        disease_class = d.get('disease_class')
        if images_dir is not None:
            image_name = os.path.join(images_dir, image_name)
        image_files.append(image_name)
        annotation_dict[image_name] = disease_class
    return image_files, annotation_dict


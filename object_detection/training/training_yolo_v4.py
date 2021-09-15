#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

"""
Contains the functionality to train a YOLOv4 network. Annotation is required in 
YOLO style but can be converted from PascalVOC as well by convert_pascal_annotations().

Adjust parameters and run the script, train the network with
train_network().

INPUT_MODEL can be a valid .weights file or a present model - in this case, the 
model weights will be loaded before fitting the present data. Can be empty as well
to train a model from scratch.
"""

import os, sys
YOLO_LIBRARY = '/home/omen6/KI-Projekt/prediction_tool_yolo/yolo-v4-tf.keras-master/'
sys.path.append(YOLO_LIBRARY)


import xml.etree.ElementTree as ET
from glob import glob

from utils import DataGenerator, read_annotation_lines
from models import Yolov4

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# specify gpu to use ("0" or "1")
GPU_TO_USE = "0"


# Declare Error level and set precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_TO_USE

tf.compat.v1.logging.set_verbosity(0)
tf.autograph.set_verbosity(0)


config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = InteractiveSession(config=config)


# CREATE ANNOTATION FILE FROM PASCAL VOC STYLE
XML_PATH = '/home/omen6/Schreibtisch/Gnu_Landau_1/Label/'
CLASSES_PATH = '/home/omen6/Schreibtisch/Gnu_Landau_1/classes.txt'
TXT_PATH = '/home/omen6/Schreibtisch/Gnu_Landau_1/annotation.txt'

# TRAINING PARAMETERS
INPUT_MODEL = '/home/omen6/KI-Projekt/Netzwerke-ObjectDetection/2020-11-16_Basisnetzwerk_Antilopen/'
VAL_SPLIT = 0.05
IMAGE_FOLDER = '/home/omen6/Schreibtisch/Gnu_Landau_1/Bilder/'
NUM_EPOCHS = 260
OUTPUT_PATH = '/home/omen6/Schreibtisch/Gnu_Landau_1/2021-03-20_OD_Gnu_Landau_1/'

FROM_EPOCH = 0
CHECKPOINT_PATH = '/home/omen6/Schreibtisch/Gnu_Landau_1/2021-03-20_OD_Gnu_Landau_1-ckpt/'

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_pascal_annotations():
    
    '''loads the classes'''
    def get_classes(classes_path):
        with open(classes_path, encoding='ISO-8859-1') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    
    classes = get_classes(CLASSES_PATH)
    assert len(classes) > 0, 'no class names detected!'
    print(f'num classes: {len(classes)}')
    
    # output file
    list_file = open(TXT_PATH, 'w+', encoding='ISO-8859-1')
    
    i = 0
    for path in glob(os.path.join(XML_PATH, '*.xml')):
        i += 1
        in_file = open(path, encoding='ISO-8859-1')
    
        # Parse .xml file
        tree = ET.parse(in_file)
        root = tree.getroot()
        # Write object information to .txt file
        file_name = path.split('/')[-1] 
        file_name = file_name.split('.')[0] + '.jpg'
        #print(file_name)
        list_file.write(file_name)
        for obj in root.iter('object'):
            cls = obj.find('name').text 
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        list_file.write('\n')
    list_file.close()
    print(str(i))



    
def train_network():
    make_directory(OUTPUT_PATH)
    make_directory(CHECKPOINT_PATH)
    
    train_lines, val_lines = read_annotation_lines(TXT_PATH, test_size=VAL_SPLIT)
    class_name_path = CLASSES_PATH
    
    print("Read training and validation lines.")
                 
    
    data_gen_val = DataGenerator(annotation_lines = val_lines, 
                                   class_name_path = class_name_path, 
                                   folder_path = IMAGE_FOLDER,
                                   max_boxes = 10,
                                   shuffle = True,
                                   augmentation = False)
    
    print("Created Validation set:" + str(len(val_lines)))
    
    data_gen_train = DataGenerator(annotation_lines = train_lines, 
                                   class_name_path = class_name_path, 
                                   folder_path = IMAGE_FOLDER,
                                   max_boxes = 10,
                                   shuffle = True,
                                   augmentation = True)
    
    print("Created Training set:" + str(len(train_lines)))
    
    if INPUT_MODEL.endswith('.weights'):
        model = Yolov4(weight_path=INPUT_MODEL, 
                       class_name_path=class_name_path)
    elif not os.path.exists(INPUT_MODEL):
        model = Yolov4(weight_path=None, 
                       class_name_path=class_name_path)
    else:
        model = Yolov4(weight_path=None, 
                       class_name_path=class_name_path)
    
    print("Model loaded.")
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model.fit(data_gen_train, 
              initial_epoch=0,
              epochs=NUM_EPOCHS, 
              val_data_gen=data_gen_val,
              callbacks=[model_checkpoint_callback])
    
    model.save_model(OUTPUT_PATH)
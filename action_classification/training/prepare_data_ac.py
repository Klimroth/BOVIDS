#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

"""
Small functions that facilitate the task of generating a training and validation set.

join_datasets():
    - input list of folders that all contain training or validation data
    (thus subfolders 0/, 1/, ... of images)
    - output folder
    - NESTED: If true, the input list is supposed to contain folders of train data.
    
    --> merges those folders by copying all files to the destination, creating a larger
    testing / training set.
    
    
validation_split():
    - input folder that contains training data (thus subfolders 0/, 1/, ... of images)
    - output folder which should contain the validation images
    - validation split percentage
    - list of classes
    --> MOVES each image with probability val_split into the validation set
    
merge_classes():
    - Input folder that contains training or validation data
    (thus subfolders 0/, 1/, ... of images)
    - Output folder which shall contain the new training or validation set
    - behavior mapping dictionary: assigns each class which percentage of images
      will be copied to which class -- '99' is used as a dismissing class
      
      --> Used to create binary classification sets.
"""

import os
import shutil
import random


############# JOIN DATASETS ######################
# merges multiple structures of the form 0/, 1/, ... to one large structure
INPUT_FOLDERS_JD = []
OUTPUT_FOLDER_JD = ''
NESTED = True


############# VALIDATION SPLIT ######################
VAL_SPLIT = 0.1
INPUT_FOLDER_VS = ''
OUTPUT_FOLDER_VS = ''

BEHAVIOR_CLASSES = ['0', '1', '2']


############# MAKE BINARY DATA ######################
INPUT_FOLDER_BD = ''
OUTPUT_FOLDER_BD = ''

BEHAVIOR_MAPPING = {'0': {'0': 1.0, 
                        '1': 0.0,
                        '99': 0.0}, 
                    '1': {'0': 0.0,
                          '1': 0.7,
                          '99': 0.3},
                    '2': {'0': 0.0,
                          '1': 0.3,
                          '99': 0.7}
                }


############ Random Upsampling #####################
UPSAMPLING_FOLDER = ''
TARGET_SIZE = 11111


############ SUBSET ################################
SUBSET_FOLDER = ''
SUBSET_OUT = ''
CLASS_SIZES_PER_IND = { '0': 250, '1': 150, '2': 250}


def get_random_subset(f = SUBSET_FOLDER, out = SUBSET_OUT, class_sizes = CLASS_SIZES_PER_IND):
    
    all_img = {}
    individuals = []
    
    for b in sorted(os.listdir(f)):
        if not b in all_img.keys():
            all_img[b] = {}
        for img in sorted(os.listdir(f + b)):
            ind = '_'.join( [img.split('_')[1], img.split('_')[2], img.split('_')[3] ] )
            if not ind in all_img[b].keys():
                all_img[b][ind] = []
            if not ind in individuals:
                individuals.append(ind)
            all_img[b][ind].append( f + b + '/' + img )
    
    real_ind = []
    for ind in sorted( individuals ):
        print( ind, [ len(all_img[x][ind]) for x in sorted(class_sizes.keys()) if ind in all_img[x].keys() ] )
        if len([ len(all_img[x][ind]) for x in sorted(class_sizes.keys()) if ind in all_img[x].keys() ]) == len(class_sizes.keys()):
            real_ind.append(ind)
    
    taken_images = {}
    for b in all_img.keys():
        taken_images[b] = []
        for ind in real_ind:
            take = random.sample( all_img[b][ind], min( class_sizes[b], len(all_img[b][ind]) ) )
            for img in take:
                taken_images[b].append( img )
    
    for b in taken_images.keys():
        dst = out + b + '/'
        _ensure_dir(dst)
        
        for img in taken_images[b]:
            img_name = img.split('/')[-1]
            shutil.copy2(img, dst + img_name)
        
        
    
    
                       

def random_upsampling(f = UPSAMPLING_FOLDER,
                      target = TARGET_SIZE):
    curr_size = len( os.listdir(f) )
    to_sample = max(0, target - curr_size)
    
    resample = random.sample( os.listdir(f), to_sample )
    
    for img in resample:
        img_base = img[:-4]
        src = f + img
        dst = f + img_base + '_copy.jpg'
        shutil.copy2(src, dst)



def _ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def get_dataset_size(counter = [0,0,0], inputf = INPUT_FOLDERS_JD, outputf = OUTPUT_FOLDER_JD, 
                  nested = NESTED):

    if nested:
        for nested_f in inputf:
            for pf in sorted(os.listdir(nested_f)):
                for j in range(len(counter)):
                    try:
                        counter[j] += len(os.listdir(nested_f + pf + '/' + str(j)))
                    except:
                        pass
    print(counter)
                


def join_datasets(inputf = INPUT_FOLDERS_JD, outputf = OUTPUT_FOLDER_JD, 
                  nested = NESTED):
    
    _ensure_dir(outputf)
    if nested:
        for nested_f in inputf:
            print(nested_f)
            for pf in sorted(os.listdir(nested_f)):
                f = nested_f + pf + '/'
                shutil.copytree(src = f, 
                        dst = outputf, 
                        dirs_exist_ok = True)
    else:
        for f in inputf:
            shutil.copytree(src = f, 
                            dst = outputf, 
                            dirs_exist_ok = True)
        
def validation_split(inputf = INPUT_FOLDER_VS, outputf = OUTPUT_FOLDER_VS, 
                     perc = VAL_SPLIT, behavs = BEHAVIOR_CLASSES):
    
    for b in behavs:
        _ensure_dir(outputf + b)
        
        to_sample = round(len(os.listdir(inputf + b))*perc)
        imgs = random.sample( os.listdir(inputf + b), to_sample)
        
        for img in imgs:
            src = inputf + b + '/' + img
            dst = outputf + b + '/' + img
            shutil.move(src = src, dst = dst)



def merge_classes(inputf = INPUT_FOLDER_BD, outputf = OUTPUT_FOLDER_BD,
                     behav_map = BEHAVIOR_MAPPING):
    
    for behav in sorted(os.listdir(inputf)):
        if not behav in behav_map.keys():
            continue
        
        for b in behav_map[behav].keys():
            if b == '99':
                continue
            _ensure_dir(outputf + b)
        
        keys, vals = [], []
        for b in behav_map[behav].keys():
            keys.append(b)
            vals.append(behav_map[behav][b])
        
        class_to_go = random.choices( population=keys, 
                           weights=vals, k=len(os.listdir(inputf + behav)))

        j = 0
        for img in os.listdir(inputf + behav):
            
            if class_to_go[j] == '99':
                j += 1
                continue
            
            src = inputf + behav + '/' + img
            dst = outputf + class_to_go[j] + '/' + img
            
            shutil.copy(src, dst)
            j += 1
            


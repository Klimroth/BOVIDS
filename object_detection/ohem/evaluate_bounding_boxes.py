#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import os
import numpy as np
import cv2
from collections import Counter
import shutil
import random



BASEPATH_INPUT = ''
ENCLOSURE_CODE = ''
TMP_LABEL_FILE = ''
MOVE_BASEPATH = ''


FLAT_IMAGE = False


PERCENTAGE_IMAGES_MOVING = 1


MOVE_LEFT = '4'
MOVE_RIGHT = '5'
END_EVALUATION = 'p'

SET_GOOD = 'a'
SET_MEDIUM = 's'
SET_BAD = 'd'
SET_SWAPPED = 'w'
SET_UNLABELLED = 'u'

COLOR_FOR_INDIVIDUALS = {
    }



##############################################################################

SPEC, ZOO, ENC = ENCLOSURE_CODE.split('_')

FOLDERPATH_IMAGES = '{}Bilder/{}/{}/{}/'.format(BASEPATH_INPUT, SPEC, ZOO, ENC)
FOLDERPATH_LABELS = '{}Label/{}/{}/{}/'.format(BASEPATH_INPUT, SPEC, ZOO, ENC)


DEST_IMAGES_GOOD = '{}gut/Bilder/{}/{}/{}/'.format(MOVE_BASEPATH, SPEC, ZOO, ENC)
DEST_LABELS_GOOD = '{}gut/Label/{}/{}/{}/'.format(MOVE_BASEPATH, SPEC, ZOO, ENC)

DEST_IMAGES_BAD = '{}bad/Bilder/{}/{}/{}/'.format(MOVE_BASEPATH, SPEC, ZOO, ENC)
DEST_LABELS_BAD = '{}bad/Label/{}/{}/{}/'.format(MOVE_BASEPATH, SPEC, ZOO, ENC)

DEST_IMAGES_SWAPPED = '{}swapped/Bilder/{}/{}/{}/'.format(MOVE_BASEPATH, SPEC, ZOO, ENC)
DEST_LABELS_SWAPPED = '{}swapped/Label/{}/{}/{}/'.format(MOVE_BASEPATH, SPEC, ZOO, ENC)




##############################################################################









def _ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def _create_blank_image(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def get_statistics(eval_file = TMP_LABEL_FILE):
    print(eval_file)
    image_dict = np.load(eval_file,allow_pickle='TRUE').item()
    evaluation_seq = [ x['evaluation'] for  key, x in image_dict.items() if key != 'current_index_showing']
    count_seq = Counter(evaluation_seq)
    good = int(count_seq[3])
    medium = int(count_seq[2])
    bad = int(count_seq[1])
    swapped = int(count_seq[9])
    no_label = int(count_seq[0])
    
    overall = good + medium + bad + swapped
    perc_good = round(100.0*good/overall, 2)
    perc_mid = round(100.0*medium/overall,2)
    perc_bad = round(100.0*bad/overall,2)
    perc_swapped = round(100.0*swapped/overall,2)

    print("Good labels: " + str( good ) + " ("+ str(perc_good) +"%)")
    print("Medium labels: " + str( medium ) + " ("+ str(perc_mid) +"%)")
    print("Bad labels: " + str( bad ) + " ("+ str(perc_bad) +"%)")
    print("Swapped labels: " + str( swapped ) + " ("+ str(perc_swapped) +"%)")
    print("No evaluation given: " + str(no_label))
    
def move_data_by_evaluation_value(eval_file = TMP_LABEL_FILE, 
                                   dest_good_images = DEST_IMAGES_GOOD,
                                   dest_good_labels = DEST_LABELS_GOOD,
                                   dest_bad_images = DEST_IMAGES_BAD,
                                   dest_swapped_images = DEST_IMAGES_SWAPPED,
                                   dest_bad_labels = DEST_LABELS_BAD,
                                   dest_swapped_labels = DEST_LABELS_SWAPPED,
                                   copy_files = True, folderpath_images = FOLDERPATH_IMAGES, 
                                   folderpath_labels = FOLDERPATH_LABELS,
                                   perc = PERCENTAGE_IMAGES_MOVING):

    _ensure_dir(dest_bad_images)
    _ensure_dir(dest_swapped_images)
    
    _ensure_dir(dest_bad_labels)
    _ensure_dir(dest_swapped_labels)

    _ensure_dir(dest_good_images)
    _ensure_dir(dest_good_labels)
    
    image_dict = np.load(eval_file,allow_pickle='TRUE').item()
    for image_name, image_info in image_dict.items():
        
        x = random.random()
        if x > perc:
            continue
        
        img_path = folderpath_images + image_name + '.jpg'
        label_path = folderpath_labels + image_name + '.xml'
                
        if image_name == 'current_index_showing':
            continue
        if image_info['evaluation'] == 3:
            if copy_files:                
                shutil.copy(img_path, dest_good_images + image_name + ".jpg")
                shutil.copy(label_path, dest_good_labels + image_name + ".xml")
            else:
                shutil.move(img_path, dest_good_images + image_name + ".jpg")
                shutil.move(label_path, dest_good_labels + image_name + ".xml")
        elif image_info['evaluation'] == 1:
            if copy_files:
                shutil.copy(img_path, dest_bad_images + image_name + ".jpg")
                shutil.copy(label_path, dest_bad_labels + image_name + ".xml")
            else:
                shutil.move(img_path, dest_bad_images + image_name + ".jpg")
                shutil.move(label_path, dest_bad_labels + image_name + ".xml")
        elif image_info['evaluation'] == 9:
            if copy_files:
                shutil.copy(img_path, dest_swapped_images + image_name + ".jpg")
                shutil.copy(label_path, dest_swapped_labels + image_name + ".xml")
            else:
                shutil.move(img_path, dest_swapped_images + image_name + ".jpg")
                shutil.move(label_path, dest_swapped_labels + image_name + ".xml")

                

    
#    if write_unlabelled:
#        image_filenames = [ f[:-4] for f in os.listdir(folderpath_images) if f.endswith(".jpg") ]
#        label_filenames = [ f[:-4] for f in os.listdir(folderpath_labels) if f.endswith(".xml") ]
#            
#        image_filenames_without_label = [ f for f in image_filenames if f not in label_filenames ]
#        
#        for image_name in image_filenames_without_label:
#            img_path = folderpath_images + image_name + ".jpg"
#            if copy_files:
#                shutil.copy(img_path, dest_images_without_label + image_name + ".jpg")
#            else:
#                shutil.move(img_path, dest_images_without_label + image_name + ".jpg")

def cv_read(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

    
def _get_boxes_from_xml(label_file):
    ret = {}
    tree = ElementTree.parse(label_file)
    root = tree.getroot()
    obj_list = root.findall('object')
    for obj in obj_list:
        name = obj.find('name')
        ind_name = name.text.rstrip("\r\n")
        boxes = []
        for box in obj.iter('bndbox'):
            
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            
        if len(boxes) > 1:
            print("Error: Multiple boxes for " + ind_name)
        if len(boxes) == 1:
            ret[ind_name] = boxes[0]
        
    return ret

def _draw_boxes_on_img(img, label_file, colors = COLOR_FOR_INDIVIDUALS):
    
    box_dict = _get_boxes_from_xml(label_file)
    for individual_name in box_dict.keys():
        if individual_name in colors.keys():
            curr_col = colors[individual_name]
        else:
            curr_col = (50, 205, 50)
        x1, y1, x2, y2 = box_dict[individual_name]
        cv2.rectangle(img, (x1,y1), (x2,y2), curr_col, 2)
    
    return img

def evaluate_folder(folderpath_images = FOLDERPATH_IMAGES, 
                     folderpath_labels = FOLDERPATH_LABELS, 
                     tmp_eval_file = TMP_LABEL_FILE, 
                     move_left = MOVE_LEFT, move_right = MOVE_RIGHT, 
                     set_bad = SET_BAD, set_medium = SET_MEDIUM, 
                     set_good = SET_GOOD, end_evaluation = END_EVALUATION,
                    set_swapped = SET_SWAPPED, set_unlabelled = SET_UNLABELLED):
    if not os.path.exists(folderpath_images):
        print(folderpath_images)
        print("Image folder not found.")
        return
    if not os.path.exists(folderpath_labels):
        print("Label folder not found.")
        print(folderpath_labels)
        return
    tmp_eval_dir = "/".join(tmp_eval_file.split("/")[:-1])
    if not os.path.exists(tmp_eval_dir):
        print("Temporary data folder not found.")
        return
    if not tmp_eval_file.endswith("npy"):
        print("Temporary file needs to be in .npy format.")
        return
    
    # load respectively extend 
    image_dict = {}
    image_dict['current_index_showing'] = 0
    
    if os.path.exists(tmp_eval_file):
        image_dict = np.load(tmp_eval_file,allow_pickle='TRUE').item()
        
    
    image_filenames = [ f[:-4] for f in os.listdir(folderpath_images) if f.endswith("jpg") ]
    label_filenames = [ f[:-4] for f in os.listdir(folderpath_labels) if f.endswith("xml") ]
        
    image_filenames_with_label = [ f for f in image_filenames if f in label_filenames ]
    
    print("Labelled images: " + str(len(image_filenames_with_label))+ ". Total images: " + str(len(image_filenames)) + "." )
    
    # remove deleted files
    # remove_entries = []
    # remove_one = False
    # for image_name, image_info in image_dict.items():
    #    if image_name == 'current_index_showing':
    #        continue
    #    if image_name not in image_filenames_with_label:
    #        remove_entries.append(image_name)
    #        remove_one = True
    #if remove_one:
    #    # save copy of labelfile
    #    x = int(10000000000*np.random.random())
    #    shutil.copy(tmp_eval_file, tmp_eval_file[:-4] + "_tmp_"+str(x)+".npy")
    #    for remove_key in remove_entries:
    #        del image_dict[remove_key]
    
    # update new files
    for filename in image_filenames_with_label:
        if filename in image_dict.keys():
            continue
        image_dict[filename] = {'imagepath': folderpath_images + filename + ".jpg",
                                'labelpath': folderpath_labels + filename + ".xml",
                                'evaluation': 0}    
        
    np.save(tmp_eval_file, image_dict)
    
    #### start evaluation process
    show_images = True
    curr_index = image_dict['current_index_showing']
    index_list = sorted([x for x in image_dict.keys() if x != 'current_index_showing'])
    len_index_list = len(index_list)
    while show_images:
        
        image_dict = np.load(tmp_eval_file,allow_pickle='TRUE').item()
        
        print("Current image: " + str(curr_index + 1) + " of " + str(len_index_list))
        
        filename = index_list[curr_index]     
        
        imgpath = folderpath_images + filename + '.jpg'
        label_path = folderpath_labels + filename + '.xml'   

        evaluation_val = image_dict[filename]['evaluation']
        #print(imgpath)
        img_frame = cv_read(imgpath)
        height, width, channels = img_frame.shape
        
        # decide color
        if evaluation_val == 1:
            eval_col = (255, 0, 0) #red (bad)
        elif evaluation_val == 2:
            eval_col = (255, 255, 0) # yellow (medium)
        elif evaluation_val == 3:
            eval_col = (0,255,0) # green (good)
        elif evaluation_val == 9:
            eval_col = (0,0,255) # blue (swapped)
        else:
            eval_col = (100,100,100) # grey
        
        img_frame = _draw_boxes_on_img(img = img_frame, label_file = label_path )
        eval_img = _create_blank_image(30, height, rgb_color=eval_col)
        img_display = np.concatenate((eval_img, img_frame, eval_img), axis=1)
        img_display = cv2.resize(img_display, (900, int(900*height/width)))
        
        cv2.imshow("Evaluate drawn boxes.", img_display)
        
        key_pressed = False
        while not key_pressed:
            key = cv2.waitKey(0)
            if key == ord(move_left):
                if curr_index == 0:
                    image_dict['current_index_showing'] = curr_index
                    #show_images = False
                else:
                    curr_index = (curr_index - 1) % len_index_list
                    image_dict['current_index_showing'] = curr_index
                np.save(tmp_eval_file, image_dict)
                key_pressed = True  
            elif key == ord(move_right):
                if curr_index == len_index_list - 1:
                    image_dict['current_index_showing'] = curr_index
                    #show_images = False
                else:
                    curr_index = (curr_index + 1) % len_index_list
                    image_dict['current_index_showing'] = curr_index
                np.save(tmp_eval_file, image_dict)
                key_pressed = True
            elif key == ord(set_swapped):
                image_dict[filename]['evaluation'] = 9
                np.save(tmp_eval_file, image_dict)
                key_pressed = True
            elif key == ord(set_good):
                image_dict[filename]['evaluation'] = 3
                np.save(tmp_eval_file, image_dict)
                key_pressed = True
            elif key == ord(set_medium):
                image_dict[filename]['evaluation'] = 2
                np.save(tmp_eval_file, image_dict)
                key_pressed = True
            elif key == ord(set_bad):
                image_dict[filename]['evaluation'] = 1
                np.save(tmp_eval_file, image_dict)
                key_pressed = True
            elif key == ord(set_unlabelled):
                image_dict[filename]['evaluation'] = 0
                np.save(tmp_eval_file, image_dict)
                key_pressed = True
            elif key == ord(end_evaluation):
                show_images = False
                key_pressed = True
    
    cv2.destroyAllWindows()
    
    # Automatisches verschieben der Daten? 
            

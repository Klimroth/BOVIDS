#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

"""
Contains the functionalities to identify individuals and cut them out.
"""

import configuration as cf
import os, shutil
from models import Yolov4
from collections import Counter

import pprint
import numpy as np
from datetime import datetime
import cv2
from operator import itemgetter

YOLO_CONFIG = {
    # Basic
    'img_size': (416,416, 3), # 416
    'anchors': [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401], #459, 401 statt 50, 170
    'strides': [8, 16, 32],
    'xyscale': [1.2, 1.1, 1.05],

    # Training
    'iou_loss_thresh': 0.5,
    'batch_size': 4,
    'num_gpu': 1,
    # Inference
    'max_boxes': 1,
    'iou_threshold': 0.5,
    'score_threshold': 0.9,
}





def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _cut_out_prediction_data(images_to_predict, 
                             od_model,  
                             enclosure_code, 
                             output_folder, 
                             label_names, 
                             img_size=cf.IMG_SIZE,
                             min_detection_score = 0.9):
    
    def _individual_name_from_boxcode(individual_codes, yolo_label):
        if len(individual_codes) == 1:
            return individual_codes[0]

        if yolo_label.startswith("Elenantilope"):
            yolo_label = yolo_label.replace("Elenantilope", "Elen")
        return yolo_label
    
    def _post_process_boxes(past_detections, curr_detection):
        # TODO: implement prediction history, for instance, tracking
        
        if len(curr_detection) > 1:
            label_box_list = sorted(curr_detection, key=itemgetter(5), reverse = True)
            if label_box_list[0][0] == label_box_list[1][0]: # both animals have the same label
                poss_names = [name for name in label_names]
                poss_names.remove(label_box_list[0][0])
                label_box_list[1][0] = poss_names[0]
                
                
            return label_box_list[0:2]
        
        return curr_detection
    
    
    i = 1
    
    previous_detections = []
    for img_path in images_to_predict:    
        
        i+= 1

        img = cv2.imread(img_path)
        h,w,c = img.shape
        
        img_name = img_path.split("/")[-1]
        interval_num = img_path.split("/")[-2] 
        
        x = od_model.predict(img_path, plot_img = False)
        
        label_box_list = []
        
        # label_box_list will be [  [label, x1, y1, x2, y2, score]   ]
        for box_index in range(len(x)):
            if x.iloc[box_index]['score'] >= min_detection_score:    
                app_list = []
                x1, x2 = x.iloc[box_index]['x1'], x.iloc[box_index]['x2']
                y1, y2 = x.iloc[box_index]['y1'], x.iloc[box_index]['y2']
                label_val = x.iloc[box_index]['class_name']
                app_list.append(label_val)
                app_list.append(x1)
                app_list.append(y1)
                app_list.append(x2)
                app_list.append(y2)
                app_list.append(x.iloc[box_index]['score'])
                label_box_list.append(app_list)
        
        if len(label_box_list) > 0:
            
            if len(label_names) > 2:
                print("ERROR: Not implemented right now (3 or more classes).")
                return
            
            if len(label_names) == 2:
                # exactly two individuals                
                label_box_list_postprcessed = _post_process_boxes(past_detections = previous_detections, 
                                                                  curr_detection = label_box_list)
                
            if len(label_names) == 1:
                label_box_list_postprcessed = [sorted(label_box_list, key=itemgetter(5), reverse = True)[0]]
            
            
                
            
            for label, x1, y1, x2, y2, score in label_box_list_postprcessed:
                box_part = img[y1:y2, x1:x2]
                box_part_rs = cv2.resize(box_part, img_size, interpolation=cv2.INTER_AREA)
                
                ind_name = _individual_name_from_boxcode(individual_codes = label_names, yolo_label=label)
                
                save_path = output_folder + ind_name + '/' + interval_num + '/'
                ensure_dir(save_path)
                cv2.imwrite(save_path + img_name, box_part_rs)
                
                text_file = open(save_path + "box_info.txt", "a+")
                text_file.write(img_name + "-" + str(y1) + "*" + str(x1) + "*" + str(y2) + "*" + str(x2) + "\n")
                text_file.close()
            
            previous_detections.append(label_box_list_postprcessed)
        else:
            previous_detections.append(False)
        if len(previous_detections) > 4:
            previous_detections = previous_detections[1:4]
        
                
                
    
    

def _predict_one_night(input_path_night, od_model, od_labels, enclosure_code, output_folder, min_conf):    
    
   
    image_path_to_predict = []
    sub_folders = [input_path_night + time_intervall + '/' for time_intervall in sorted(os.listdir(input_path_night))]
    for sub_fold in sub_folders:
        images_in_sub = [sub_fold + img_name for img_name in sorted(os.listdir(sub_fold))]
        for img_path in sorted(images_in_sub):
            image_path_to_predict.append(img_path)
   
   
    print('Night contains %d images.' % len(image_path_to_predict))
    _cut_out_prediction_data(image_path_to_predict, od_model, enclosure_code, output_folder, od_labels, min_detection_score=min_conf)


 
        


def merge_timeinterval_images(datum, path_to_intervalfolders, output_folder_base):
    
        
    """
    

    Parameters
    ----------
    path_to_intervalfolders : string
        TMP_STORAGE/intervals/enclosure_code/datum/
        contains for each individual a folder, each of those contains folders of intervals
    output_folder_base: string
        TMP_STORAGE/

    Returns
    -------
    Writes output_path_intervalimg/interval_num.jpg and  output_path_single_frame/frame_num.jpg (up to 2 out of 4)

    """
    
    def _write_joint_image(imgpath_list, out_directory, time_interval):
        img_list = []
        for imgpath in imgpath_list:
            img = cv2.imread(imgpath)
            img_list.append(img)
        
        if len(img_list) == 0:
            return
        
        h, w, d = img_list[0].shape
        
        img_black = np.zeros([w, h, d],dtype=np.uint8)
        if len(img_list) == 1:
            vis1 = np.concatenate((img_list[0], img_black), axis=1)            
            vis2 = np.concatenate((img_black, img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
                
        elif len(img_list) == 2:
            vis1 = np.concatenate((img_list[0], img_list[1]), axis=1)            
            vis2 = np.concatenate((img_black, img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
        
        elif len(img_list) == 3:
            vis1 = np.concatenate((img_list[0], img_list[1]), axis=1)            
            vis2 = np.concatenate((img_list[2], img_black), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
        
        elif len(img_list) == 4:
            vis1 = np.concatenate((img_list[0], img_list[1]), axis=1)            
            vis2 = np.concatenate((img_list[2], img_list[3]), axis=1)
            vis = np.concatenate((vis1, vis2), axis=0)
        
        vis = cv2.resize(vis, (h, w), interpolation=cv2.INTER_AREA)
        
        ensure_dir(out_directory)
        cv2.imwrite(out_directory + str(time_interval).zfill(7) + ".jpg", vis)
    
    def _write_single_frame(imgpath_list, out_directory):
        
        img_list = []
        img_names = []
        ensure_dir(out_directory)
        
        for imgpath in imgpath_list:
            img = cv2.imread(imgpath)
            img_list.append(img)
            name = imgpath.split("/")[-1]
            img_names.append(name)
        
        if len(img_list) == 0:
            return
        
        if len(img_list) == 1:
            cv2.imwrite(out_directory + img_names[0], img_list[0])
                
        elif len(img_list) in [2, 3]:
            cv2.imwrite(out_directory + img_names[0], img_list[0])
            cv2.imwrite(out_directory + img_names[1], img_list[1])
        
        elif len(img_list) == 4:
            cv2.imwrite(out_directory + img_names[0], img_list[0])
            cv2.imwrite(out_directory + img_names[2], img_list[2])
        
            
    if not os.path.exists(path_to_intervalfolders + datum):
        print("Warning: No predicted images. Was the individual always out?")
        return
    
    for individual_name in os.listdir(path_to_intervalfolders + datum + '/'):
        
        
        output_path_intervalimg = output_folder_base + individual_name + '/multiple_frames/' + datum + '/'                                  
        output_path_single_frame = output_folder_base + individual_name + '/single_frames/'  + datum + '/'
        output_path_text = output_folder_base + individual_name + '/position_files/'  + datum + '/'
                                          
        for time_interval in os.listdir(path_to_intervalfolders + datum + '/' + individual_name + "/" ):
            curr_path = path_to_intervalfolders + datum + '/' + individual_name + "/" + time_interval + "/"
            imgpath_list = [curr_path + f for f in os.listdir(curr_path) if f.endswith("jpg")]            
            position_file = [curr_path + f for f in os.listdir(curr_path) if f.endswith("txt")]
            
            out_directory_joint = output_path_intervalimg + '/0/'        
            _write_joint_image(imgpath_list, out_directory_joint, time_interval)
            
            out_directory_single = output_path_single_frame + '/0/'
            _write_single_frame(imgpath_list, out_directory_single)
            
            # write text document with the position information
            if len(position_file):
                position_file = position_file[0]
                out_directory_text = output_path_text + '/'
                ensure_dir(out_directory_text)
                shutil.copy(position_file, out_directory_text + time_interval.zfill(7) + '.txt')



def predict_multiple_nights(enclosure_info, individual_info, 
                            input_path = cf.TMP_STORAGE_IMAGES, 
                            output_folder_base = cf.TMP_STORAGE_CUTOUT):
    """
    

    Parameters
    ----------
    input_path : string
        Path to .../enclosure_code/, contains the dates (folders) to predict        
    species : TYPE
        DESCRIPTION.
    zoo : TYPE
        DESCRIPTION.
    enclosure_num : TYPE
        DESCRIPTION.
    output_folder_base : string
        Folder of the form .../, will create 
            a subfolder individual_code/multiple_frames/date/0/interval_num.jpg
            a subfolder individual_code/single_frames/date/0/frame_num.jpg
            a subfolder individual_code/position_files/date/interval_num.txt

    Returns
    -------
    None.

    """

    for enclosure_code in enclosure_info.keys():
    
                
        index = -1
        modelpaths = {}

        for date in enclosure_info[enclosure_code]['dates']:
            index += 1
            
            individual_codes = enclosure_info[enclosure_code]['individuals'][index]
            enclosure_individual_code_string = enclosure_code + '_' + '+'.join([x.split("_")[2] for x in individual_codes])
        
            net_path, od_label_file = cf.get_object_detection_network(enclosure_code, 
                                                                      enclosure_individual_code_string,
                                                                      cf.BASE_OD_NETWORK,
                                                                      cf.OD_NETWORK_LABELS)
                       
                        
            if not net_path or not od_label_file:
                print('Network and / or Labels are not found for enclosure {}'.format(enclosure_individual_code_string))
                continue
        
            with open(od_label_file) as f:
                od_classes = [i.strip() for i in f.readlines()]
                
        
            if len(od_classes) != len(individual_codes):
                print("ERROR: Number of OD classes does not fit!")
                continue
        
            if len(od_classes) == 1:
                od_classes = individual_codes            
            else:
                if not Counter(od_classes) == Counter(individual_codes):
                    print("ERROR: OD Network contains different Individualcodes than expected.")
                    continue
            
            if not enclosure_individual_code_string in modelpaths.keys():
                modelpaths[enclosure_individual_code_string] = []
                        
            modelpaths[enclosure_individual_code_string].append([date, od_label_file, od_classes, net_path])
            
            

        for enclosure_individual_code_string in modelpaths:
            od_label_file, od_classes, net_path = modelpaths[enclosure_individual_code_string][0][1], modelpaths[enclosure_individual_code_string][0][2], modelpaths[enclosure_individual_code_string][0][3]
            possible_dates = [ x[0] for  x in modelpaths[enclosure_individual_code_string] ]
            
            
            YOLO_CONFIG['iou_threshold'] = cf.get_iou_thresh(enclosure_individual_code_string)
            YOLO_CONFIG['score_threshold'] = cf.get_detection_score(enclosure_individual_code_string)
            YOLO_CONFIG['max_boxes'] = len(od_classes)    
            model = Yolov4(class_name_path=od_label_file, config = YOLO_CONFIG)
            model.load_model(net_path)
        
            print("Network: " + net_path)
            print("Possible labels: " + str(od_classes))
            
            for datum in sorted(os.listdir(input_path + enclosure_code)):
                if not datum in possible_dates:
                    continue
                
                print(datetime.now().strftime('%Y-%m-%d %H:%M'), '- Object Detection: Enclosure and Individuals {} - Date {}'.format(enclosure_individual_code_string, datum))
                _predict_one_night(input_path + enclosure_code + '/' + datum + '/', 
                                   model, 
                                   od_classes, 
                                   enclosure_code, 
                                   output_folder_base + 'intervals/' + enclosure_code + '/' + datum +'/', 
                                   YOLO_CONFIG['score_threshold'])
        
                merge_timeinterval_images(datum = datum,
                                          path_to_intervalfolders = output_folder_base + 'intervals/' + enclosure_code + '/', 
                                          output_folder_base = output_folder_base)

        shutil.rmtree(output_folder_base + 'intervals/' + enclosure_code, ignore_errors = True)
    
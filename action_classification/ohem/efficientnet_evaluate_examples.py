#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

"""

evaluate()
stats()
move()

"""

import os
import numpy as np
import pandas as pd
import cv2
import shutil
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

FOLDERPATH_IMAGES = '' 
FOLDERPATH_CSV = ''
INDIVIDUAL_CODE_LIST = []


START_INDEX = 0

DEST_IMAGES = ''
KI_CUTOUT = ''


NEXT_INDIVIDUAL = 'n'
PREV_INDIVIDUAL = 'b'

MOVE_LEFT = '4'
MOVE_RIGHT = '5'
END_EVALUATION = 'p'

SET_STANDING = 'w'
SET_LYING = 'a'
SET_SLEEPING = 's'
SET_ERROR = 'f'

NEXT_IMAGE_IF_KEY = False
DISPLAY_WIDTH = 1024

COLORS = {0:(0,0,128), 1: (34,139,34), 2: (255,255,0), 3: (255,69,0), 99: (128,128,128)}



def _ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def _create_blank_image(width, height, ai_sf, ai_mf, human):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image_ai = np.zeros((height, width//2, 3), np.uint8)
    image_ai[:] = tuple(reversed(ai_sf))
    
    # Create black blank image
    image_ai_mf = np.zeros((height, width//2, 3), np.uint8)
    image_ai_mf[:] = tuple(reversed(ai_mf))
    
    image_human = np.zeros((height, width//2, 3), np.uint8)
    image_human[:] = tuple(reversed(human))
    
    image = np.concatenate((image_human, image_ai_mf), axis=1)
    image2 = np.concatenate((image_ai, image_human), axis=1)
    return image, image2



def stats(folderpath_csv = FOLDERPATH_CSV,  individual_codes = INDIVIDUAL_CODE_LIST):
    if not os.path.exists(folderpath_csv):
        print("CSV folder not found.")
        print(folderpath_csv)
        return
    
    for individual_code in individual_codes:
        print("********************************************")
        print(individual_code)
        print("********************************************")
        csv_file_path = folderpath_csv + individual_code + '.csv'
        if not os.path.exists(csv_file_path):
            print("Error (File):", csv_file_path)
            
        csvf = pd.read_csv(csv_file_path)
        ai_labels_sf = csvf['label_ai'].tolist()
        ai_labels_mf = csvf['label_mf'].tolist()
        human_labels = csvf['label_human'].tolist()
        
        print('************* Single Frame *************')
        precision, recall, fscore, _ = precision_recall_fscore_support(human_labels, ai_labels_sf, labels=[0,1,2,3,99])
        acc = accuracy_score(human_labels, ai_labels_sf)
        cm = confusion_matrix(y_true = human_labels, y_pred = ai_labels_sf, labels=[0,1,2,3,99])
        print('Accuracy: {} and F-Score: {}'.format(acc, fscore))
        print('Confusion Matrix')
        print(cm)
        
        print('************* Multiple Frame *************')
        precision, recall, fscore, _ = precision_recall_fscore_support(human_labels, ai_labels_mf, labels=[0,1,2,3,99])
        acc = accuracy_score(human_labels, ai_labels_mf)
        cm = confusion_matrix(y_true = human_labels, y_pred = ai_labels_mf, labels=[0,1,2,3,99])
        print('Accuracy: {} and F-Score: {}'.format(acc, fscore))
        print('Confusion Matrix')
        print(cm)

def move(folderpath_images = KI_CUTOUT, folderpath_csv = FOLDERPATH_CSV, 
                individual_codes = INDIVIDUAL_CODE_LIST, imagedes = DEST_IMAGES):
    
    if not os.path.exists(folderpath_images):
        print(folderpath_images)
        print("Image folder not found.")
        return
    if not os.path.exists(folderpath_csv):
        print("CSV folder not found.")
        print(folderpath_csv)
        return
    
    for individual_code in individual_codes:
        csv_file_path = folderpath_csv + individual_code + '.csv'
        print(csv_file_path)
        if not os.path.exists(csv_file_path):
            print("Error (File):", csv_file_path)
            
        csvf = pd.read_csv(csv_file_path)
        
        current_row = 0
        for current_row in range(len(csvf)):
            human_behav = csvf.loc[current_row, 'label_human']
            rel_p_img = csvf.loc[current_row, 'relative_path_cutout']
            rel_p_mf = csvf.loc[current_row, 'relative_path_mf']
            
            img_name = csvf.loc[current_row, 'identifier_sf'] 
            date = img_name.split('_')[0]
            individual_code = csvf.loc[current_row, 'individual']
            img_name_mf = csvf.loc[current_row, 'identifier_mf'] 
            
            src = folderpath_images + rel_p_img        
            src_mf = folderpath_images + rel_p_mf 
            
            if  os.path.isfile(src):            
                dst = '{}single_frames/{}/{}/{}'.format(imagedes, individual_code, human_behav, img_name) 
                _ensure_dir(dst)        
                shutil.copy2(src, dst)
                
            if  os.path.isfile(src_mf):            
                dst_mf = '{}multiple_frames/{}/{}/{}'.format(imagedes, individual_code, human_behav, img_name_mf) 
                _ensure_dir(dst_mf)        
                shutil.copy2(src_mf, dst_mf)
        
        
        

   

def cv_read(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img





def evaluate( individual_code_list = INDIVIDUAL_CODE_LIST,
        folderpath_images = FOLDERPATH_IMAGES, 
        folderpath_csv= FOLDERPATH_CSV,                                       
        move_left = MOVE_LEFT, move_right = MOVE_RIGHT, 
        set_standing = SET_STANDING, set_lying = SET_LYING, 
        set_sleeping = SET_SLEEPING, set_error = SET_ERROR, 
        end_evaluation = END_EVALUATION,
        colors = COLORS,
        start_index = START_INDEX,
        next_if_key = NEXT_IMAGE_IF_KEY,
        display_width = DISPLAY_WIDTH,
        next_individual = NEXT_INDIVIDUAL,
        prev_individual = PREV_INDIVIDUAL):
    
    
    def _load_csv(csv_file_path, start_index = start_index):
            
            if not os.path.exists(csv_file_path):
                print("Error (File):", csv_file_path)
            csvf = pd.read_csv(csv_file_path)
            
            max_row_num = len(csvf) - 1
    
            if start_index >= 1 and start_index <= max_row_num:
                current_row_num = start_index
            else:
                current_row_num = 0
            
        
            skip = 0
            if next_if_key:
                skip = 1
            
            return csvf, max_row_num, current_row_num, skip
        
    
    
    if not os.path.exists(folderpath_images + 'SingleFrame'):
        print(folderpath_images)
        print("Image folder not found (SF).")
        return
    if not os.path.exists(folderpath_images + 'MultipleFrame'):
        print(folderpath_images)
        print("Image folder not found (MF).")
        return
    if not os.path.exists(folderpath_csv):
        print("CSV folder not found.")
        print(folderpath_csv)
        return
    
    
    


    
            
    #### start evaluation process
    curr_individual = 0
    show_images = True
    while show_images:
        
        csv_file_path = folderpath_csv + individual_code_list[curr_individual] + '.csv'
        csvf, max_row_num, current_row_num, skip = _load_csv(csv_file_path)
        individual_code = individual_code_list[curr_individual]
        
        print(individual_code + ": " + str(current_row_num) + " of " + str(max_row_num))
        ai_behav = csvf.loc[current_row_num, 'label_ai']
        ai_behav_mf = csvf.loc[current_row_num, 'label_mf']
        human_behav = csvf.loc[current_row_num, 'label_human']
        img_name = csvf.loc[current_row_num, 'identifier_sf']
        img_name_mf = csvf.loc[current_row_num, 'identifier_mf']
        
        img_path = '{}SingleFrame/{}/{}/{}'.format(folderpath_images, individual_code, ai_behav, img_name)
        img_path_mf = '{}MultipleFrame/{}/{}/{}'.format(folderpath_images, individual_code, ai_behav_mf, img_name_mf)
        if not os.path.exists(img_path):
            current_row_num += 1
            print("Error: Image does not exist:", img_path)
            continue
        if not os.path.exists(img_path_mf):
            current_row_num += 1
            print("Error: Image does not exist:", img_path_mf)
            continue
        
        img_frame = cv_read(img_path)
        img_frame_mf = cv_read(img_path_mf)
        height, width, channels = img_frame.shape
        
        # decide color
        ai_col = colors[ai_behav]
        ai_col_mf = colors[ai_behav_mf]
        human_col = colors[human_behav]

        eval_img, eval_img_2 = _create_blank_image(30, height, ai_sf = ai_col, ai_mf = ai_col_mf, human = human_col)
        img_display = np.concatenate((eval_img_2, img_frame, img_frame_mf, eval_img), axis=1)
        img_display = cv2.resize(img_display, (display_width, int(0.5*display_width*height/width)))
        
        cv2.imshow("Label behavior.", img_display)
        
        key_pressed = False
        while not key_pressed:
            to_save = False
            key = cv2.waitKey(0)
            if key == ord(move_left):
                if current_row_num <= 0:
                    pass
                else:
                    current_row_num -= 1
                key_pressed = True  
            elif key == ord(move_right):
                if current_row_num == max_row_num:
                    pass
                else:
                    current_row_num += 1
                key_pressed = True
            elif key == ord(set_standing):
                csvf.loc[current_row_num, 'label_human'] = 0
                current_row_num = min(current_row_num + skip, max_row_num)
                to_save = True
                key_pressed = True
            elif key == ord(set_lying):
                csvf.loc[current_row_num, 'label_human'] = 1
                current_row_num = min(current_row_num + skip, max_row_num)
                to_save = True
                key_pressed = True
            elif key == ord(set_sleeping):
                csvf.loc[current_row_num, 'label_human'] = 2
                current_row_num = min(current_row_num + skip, max_row_num)
                to_save = True
                key_pressed = True
            elif key == ord(set_error):
                csvf.loc[current_row_num, 'label_human'] = 3
                current_row_num = min(current_row_num + skip, max_row_num)
                to_save = True
                key_pressed = True
            elif key == ord(next_individual):
                curr_individual = min(len(individual_code_list)-1, curr_individual + 1)
                key_pressed = True
            elif key == ord(prev_individual):
                curr_individual = max(0, curr_individual - 1)
                key_pressed = True
            elif key == ord(end_evaluation):
                show_images = False
                key_pressed = True
            if to_save:
                csvf.to_csv(csv_file_path, index = False)
    
    cv2.destroyAllWindows()
    

            

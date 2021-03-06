#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

GLOBAL_CONFIGURATION_PATH = ''
YOLO_LIBRARY = ''

import sys
sys.path.append(GLOBAL_CONFIGURATION_PATH)
sys.path.append(YOLO_LIBRARY)

import numpy as np
import os
import csv
import cv2
from datetime import datetime
from global_configuration import VIDEO_ORDER_PLACEMENT as BOX_PLACEMENT
from global_configuration import VIDEO_BLACK_REGIONS as POLYGON_ENDPOINTS
from global_configuration import BEHAVIOR_NAMES_BASE
from global_configuration import OD_NETWORK_LABELS_GLOBAL as GLOBAL_LABELS
from global_configuration import BASE_OD_NETWORK_GLOBAL as GLOBAL_NETS
from global_configuration import get_object_detection_network

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from operator import itemgetter
from models import Yolov4

##############################################################################

CREATE_IMAGES = True
CSV_OVERVIEW_FILE = '' # CSV File that contains the information which data should be used for training, or prepared for prediction.
BASE_PATH_TO_DATA = ''  # Contains the starting point of navigation to the videos 
BASE_PATH_TO_LABELS = '' #... and CSV-label files.

VIDEO_LEN_SPECIAL = {}
OUTPUT_PATH_MULTIPLE_FRAME = ''

##############################################################################

CREATE_YOLO_CUTOUT = True
YOLO_OUTPUT_FOLDER = ''
YOLO_NETPATH = ''

MAX_DETECTIONS = 1
MIN_CONFIDENCY = 0.9


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.logging.set_verbosity(0)
tf.autograph.set_verbosity(0)


config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = InteractiveSession(config=config)
##############################################################################

CSV_DELIMITER_OVERVIEW = ',' #On the which data use to train file
ANIMAL_NUM_SEPERATOR = ';' # The seperator in the above csv-file.
END_OF_CSV = "pred"

ALL_BEHAVIORS = {"Stehen": 0, "Liegen": 1, "Schlafen": 2, "Out": 3}
POSSIBLE_BEHAVIORS = {"Stehen": 0, "Liegen": 1, "Schlafen": 2, "Out": 3}
BEHAVIOR_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3}

INTERVAL_LEN = 7
IMAGES_PER_INTERVAL = 4

WEIGHT_FACTOR = 2.0 # Takes every n-th image per class
CSV_DELIMITER_LABELLING = ',' # Delimiting sign of the labelling csv files.




def get_yolo_net(enclosure_code, 
                 glob_nets = GLOBAL_NETS, 
                 glob_labels = GLOBAL_LABELS, 
                 path = YOLO_NETPATH):
    
    net, label = get_object_detection_network(enclosure_code = enclosure_code, 
                    enclosure_individual_code = 'not implemented',
                     basenets = glob_nets, 
                     labels = glob_labels)
    
    
    return path + net, path + label

def _check_consistency_of_behaviorlists():
    if len( [x for x in ALL_BEHAVIORS.values() if x not in BEHAVIOR_MAPPING.keys()] ) > 0:
        print("Error: Some behaviors exist but are not mapped.")
        return False
    poss_problems = [ x for x in ALL_BEHAVIORS.values() if x not in POSSIBLE_BEHAVIORS.values() ]
    real_problems = [x for x in poss_problems if BEHAVIOR_MAPPING[x] not in POSSIBLE_BEHAVIORS.values()]
    if len(real_problems) > 0:
        print("Error: Some behaviors are mapped to impossible behaviors.")
        return False

    ret = False
    if(all(x in ALL_BEHAVIORS for x in POSSIBLE_BEHAVIORS)): 
        ret = True
    if not ret:
        print("Error: Possible behaviors is not a subset on all behaviors.")

    return ret

def get_csv_label_file(species, zoo, individual_num, date, base_path = BASE_PATH_TO_LABELS):
    path = base_path+species+"/"+zoo+"/Auswertung/Boris_KI/csv-Dateien/"+date+"_"+species+"_"+zoo+"_"+str(individual_num)+"_SUM-"+str(INTERVAL_LEN)+"s_"+END_OF_CSV+".csv"
    if not os.path.exists(path):
        print("Error: "+path+" was not found.")
        return ""
    return path

def get_videofile_list(species, zoo, videolist, date, base_path = BASE_PATH_TO_DATA):
    vid_list = []
    for vid_num in videolist:
        vid_path = base_path+species+"/"+zoo+"/Videos/"+species+"_"+str(vid_num)+"/"+_correct_date(date)+"_"+species+"_"+zoo+"_"+str(vid_num)+".avi"
        if not os.path.exists(vid_path):
            print("Error: "+vid_path+" was not found.")
            return []
        vid_list.append(vid_path)
    return vid_list

def _correct_date(date):
    if not "." in date:
        return date
    return date.split(".")[2] + "-" + date.split(".")[1] + "-" + date.split(".")[0]
    
def _create_video_label_mapping(overview_file = CSV_OVERVIEW_FILE, delim = CSV_DELIMITER_OVERVIEW):
    """
    Returns a dictionary {Art_Zoo_Enclosure_Num: {individual_number: [ [video_list_per_day] ], [csv_label_list] } }
    """
    return_dict = {}
    if not os.path.exists(overview_file):
        print("Error: Overview-CSV-file was not found.")
        return return_dict

    with open(overview_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                if len(row) != 6:
                    print("Overview file has the wrong format.")
                    return return_dict
            else:
                date = _correct_date(row[0])
                species = row[1]
                zoo = row[2]
                enclosure_num = row[3]
                video_nums = row[4].split(ANIMAL_NUM_SEPERATOR)
                individual_nums = row[5].split(ANIMAL_NUM_SEPERATOR)
                for ind_num in individual_nums:
                    
                    csv_label_file = get_csv_label_file(species, zoo, ind_num, date)
                    avi_video_filelist = get_videofile_list(species, zoo, video_nums, date)
                    
                    if len(csv_label_file) < 1:
                        continue               
                    if len(avi_video_filelist) < 1:
                        continue

                    dict_key = species+"_"+zoo+"_"+str(enclosure_num)
                    if not dict_key in return_dict.keys():
                        return_dict[dict_key] = {ind_num: [ [avi_video_filelist], [csv_label_file]]}
                    elif not ind_num in return_dict[dict_key].keys():
                        return_dict[dict_key][ind_num] = [ [avi_video_filelist], [csv_label_file] ]
                    else:
                        return_dict[dict_key][ind_num][0].append(avi_video_filelist)
                        return_dict[dict_key][ind_num][1].append(csv_label_file)

            line_count += 1

    return return_dict


def _create_videolist_for_prediction(overview_file = CSV_OVERVIEW_FILE, delim = CSV_DELIMITER_OVERVIEW):
    """
    Returns a dictionary {Art_Zoo_Enclosure_Num: [video_list_per_day] }
    """
    return_dict = {}
    if not os.path.exists(overview_file):
        print("Error: Overview-CSV-file was not found.")
        return return_dict

    with open(overview_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                if len(row) != 6:
                    print("Overview file has the wrong format.")
                    return return_dict
            else:
                date = row[0]
                species = row[1]
                zoo = row[2]
                enclosure_num = row[3]
                video_nums = row[4].split(ANIMAL_NUM_SEPERATOR)
                    
                avi_video_filelist = get_videofile_list(species, zoo, video_nums, date)
           
                if len(avi_video_filelist) < 1:
                    continue

                dict_key = species+"_"+zoo+"_"+str(enclosure_num)
                if not dict_key in return_dict.keys():
                    return_dict[dict_key] = [ avi_video_filelist ]
                else:
                    return_dict[dict_key].append(avi_video_filelist)

            line_count += 1

    return return_dict

def _get_labelling_dist_from_csv(csv_filename, cut_off, delim = CSV_DELIMITER_LABELLING, 
                                 all_behaviors = ALL_BEHAVIORS):
    """
    Input: csv-labelling file.
    Output: {behavior code: amount of occurencies}

    Requires csv-file of format
    Time_Interval || Start_Frame || End-Frame || Behavior 1 || ... || Behavior n
    """
    ret_dict = {}
    for behav_code in all_behaviors.values():
        ret_dict[behav_code] = 0
    if not os.path.exists(csv_filename):
        print("Error: CSV-file was not found:"+csv_filename)
        return ret_dict

    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0

        row_to_behavior_mapping = {} #row_index: behavior_code

        for row in csv_reader:
            if line_count > cut_off:
                break
            if line_count == 0:
                if len(row) != 3 + len(all_behaviors.keys()):
                    print("Error: CSV-file has wrong fromat:"+csv_filename)
                    return ret_dict
                for j in range(3, len(row)):
                    behav_name = row[j]
                    if not behav_name in all_behaviors.keys():
                        print("Error: CSV-file contains unknown behavior:"+behav_name)
                        return ret_dict
                    behav_code = all_behaviors[behav_name]
                    row_to_behavior_mapping[j] = behav_code
            else:
                int_row = list(map(int, row))
                shown_behav = row_to_behavior_mapping[int_row[3:].index(1)+3]
                ret_dict[shown_behav] += 1
            line_count += 1
    return ret_dict
    

def get_cutoff(enclosure_code):
    species, zoo, enc = enclosure_code.split('_')
    if zoo in VIDEO_LEN_SPECIAL.keys():
        cut_off = int(VIDEO_LEN_SPECIAL[zoo]*3600 / INTERVAL_LEN)
    else:
        cut_off = int(14*3600 / INTERVAL_LEN)
    return cut_off


def _get_labelling_distribution_per_individual(video_label_map_dict, all_behaviors = ALL_BEHAVIORS, 
                                               csv_delim_labelling = CSV_DELIMITER_LABELLING):
    """
    Input: {Enclosure_Code: {Individual: [ [[list of vids]], [list of labelling files] ]}}
    Output: {Enclosure_Code: {Individual: {behav_code: amount_of_intervals}} }
    """
    ret_dict = {}
    for enc_code in video_label_map_dict.keys():
        
        cut_off = get_cutoff(enc_code)
        
        dict_of_ind = video_label_map_dict[enc_code]
        ret_dict[enc_code] = {}
        for ind_num in dict_of_ind.keys():
            # Initialise dictionary for specific individual
            ret_dict[enc_code][ind_num] = {}
            for behav_code in all_behaviors.values():
                ret_dict[enc_code][ind_num][behav_code] = 0                

            
            labelling_files = dict_of_ind[ind_num][1]
            for csv_filename in labelling_files:
                label_dist_file = _get_labelling_dist_from_csv(csv_filename = csv_filename,
                                                               delim = csv_delim_labelling,
                                                               all_behaviors = all_behaviors,
                                                               cut_off = cut_off)
                for behav_code in all_behaviors.values():
                    ret_dict[enc_code][ind_num][behav_code] += label_dist_file[behav_code]

    return ret_dict

def _get_out_code(all_behaviors = ALL_BEHAVIORS):
    if "Out" not in all_behaviors.keys():
        print("Error: One possible behavior needs to be Out.")
        return -1
    return all_behaviors["Out"]
    
    
def _map_labellings_to_real_labels(individual_label_overview, behavior_map = BEHAVIOR_MAPPING, 
                                   all_behaviors = ALL_BEHAVIORS, possible_behaviors = POSSIBLE_BEHAVIORS):
    """
    Input: {Enclosure_Code: {Individual: {behav_code: amount_of_intervals}} }
    Output: {Enclosure_Code: {Individual: {behav_code: amount_of_intervals}} } where laufen is mapped to active and so on
    """
    ret_dict = {}
    for enclosure_code in individual_label_overview.keys():
        ret_dict[enclosure_code] = {}
        for ind_num in individual_label_overview[enclosure_code]:

            ret_dict[enclosure_code][ind_num] = {}
            for poss_behav in possible_behaviors.values():
                ret_dict[enclosure_code][ind_num][poss_behav] = 0
            
            for behav_code in all_behaviors.values():
                real_behav = behavior_map[behav_code]
                ret_dict[enclosure_code][ind_num][real_behav] += individual_label_overview[enclosure_code][ind_num][behav_code]
    return ret_dict

def _get_labelling_dist(label_dict, possible_behaviors = POSSIBLE_BEHAVIORS, weight_factor = WEIGHT_FACTOR ):
    """
    Input: {Enclosure_Code: {Individual: {behav_code: amount_of_intervals}} }
    Output: {Enclosure_Code: {Individual: {behav_code: frame modolus}} } 
    """
    ret_dict = {}
    for enclosure_code in label_dict.keys():
        ret_dict[enclosure_code] = {}
        for ind_num in label_dict[enclosure_code]:
            ret_dict[enclosure_code][ind_num] = _get_labelling_dist_one_ind(ind_labeldict = label_dict[enclosure_code][ind_num],
                                                                            possible_behaviors = possible_behaviors, weight_factor = weight_factor)
    return ret_dict

def _get_labelling_dist_one_ind(ind_labeldict, possible_behaviors = POSSIBLE_BEHAVIORS, weight_factor = WEIGHT_FACTOR, sparsity = True): 
    """
    If sparsity is set and the maximum is lower than 5, we will multiply by a factor to decrease image amount.
    Input: {behav_code: amount_of_intervals}
    Output: {behav_code: frame modolus} 
    """
    ind_labeldict_without_out = {}
    for behav_code in sorted(ind_labeldict.keys()):
        if not behav_code == _get_out_code(possible_behaviors):
            ind_labeldict_without_out[behav_code] = ind_labeldict[behav_code]

    min_val = min(ind_labeldict_without_out.items(), key=lambda x: x[1])[1]
    min_key = min(ind_labeldict_without_out.items(), key=lambda x: x[1])[0]
    
    
    if min_val == 0:
        print("Error: A behavior was observed zero times at an individual. Training might be strange.")
        print("Nevertheless: I will take every 100th picture of this individual.")
        for behav_code in ind_labeldict_without_out.keys():
                ind_labeldict_without_out[behav_code] = 100
        return  ind_labeldict_without_out
    for behav_code in ind_labeldict_without_out.keys():
        if behav_code != min_key:
            ind_labeldict_without_out[behav_code] = int( np.floor(ind_labeldict_without_out[behav_code]*1.0*weight_factor / min_val) )
        else:
            ind_labeldict_without_out[behav_code] = 1.0*weight_factor
    
    max_val = max(ind_labeldict_without_out.items(), key=lambda x: x[1])[1]
    correction_factor = 1
    if max_val < 5:
        if max_val == 1:
            correction_factor = 6
        elif max_val == 2:
            correction_factor = 3
        elif max_val == 3:
            correction_factor = 2
        else:
            correction_factor = 1.5
            
    for behav_code in ind_labeldict_without_out.keys():
        ind_labeldict_without_out[behav_code] = int( np.ceil(ind_labeldict_without_out[behav_code]*correction_factor) )
        
    return ind_labeldict_without_out

def _get_labelling_sequence(csv_filename, cut_off, interval_len=INTERVAL_LEN, 
                            behav_map = BEHAVIOR_MAPPING, map_behavior = True, delim = CSV_DELIMITER_LABELLING,
                            get_intervals = False):
    """
    Input: csv_file which contains label sequence
    Requires csv-file of format
    Time_Interval || Start_Frame || End-Frame || Behavior 1 || ... || Behavior n
    Parameter: map_behavior; if activated, 
    Output: sequence of behavioral categories
    """
    ret_list = []
    
    if not os.path.exists(csv_filename):
        print("Error: CSV-file was not found:"+csv_filename)
        return ret_list

    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
  
        for row in csv_reader:
            if line_count > 0:
                int_row = list(map(int, row))
                shown_behav = int_row[3:].index(1)

                if map_behavior:
                    shown_behav = behav_map[shown_behav]

                ret_list.append(shown_behav)
                
            line_count += 1
            if line_count > cut_off:
                break
    if not get_intervals:      
        return np.repeat(ret_list, interval_len)
    
    return ret_list

def _decide_width_height(width_dims, height_dims, amount_streams):
    is_low_res = False
    ratios = [width_dims[i]*1.0/height_dims[i] for i in range(len(width_dims))]
    #low res: 1.333
    #high res: 1.777 oder 1.666
    if min(ratios) < 1.4:
        is_low_res = True

    res = (0, 0)


    # 1 image
    # only one video_file (no arrangement necessary)
    # 1280, 720 (hd), 800, 600 (low res)
        
    # 2 images
    # there will be exactly two pictures side by side
    # 1280 x 360 (hd), 1280 x 480 (low res)
        
    # 3 or 4 images
    # square of pictures with one (zero) black frame
    # 1280 x 720 (hd), 1280 x 960 (low res)

    # 5 or 6 images
    # first row 3, second row 2 + one (no) black frame
    # 1278 x 720 (hd), 1278 x 480 (low res)
    if amount_streams == 1:
        if is_low_res:
            res = (800, 600)
        else:
            res = (1280, 720)    
    elif amount_streams in [2,3,4]:        
        if is_low_res:
            res = (640, 480)
        else:
            res = (640, 360)
    elif amount_streams in [5,6]:
        if is_low_res:
            res = (426, 320)
        else:
            res = (426, 240)
    else:
        print("Error: It is currently not supported to have more than 6 video streams!")
        return False, res
    return True, res


def _save_image(frame, path, filename):
    """
    Saves the np.array() frame as an image to path. 
    """
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(path+"/"+filename, frame) 
    

def  _concatenate_frames(enclosure_code, enclosure_video_code, frames, res, amount_streams):
    """
    Input: Enclosure_code, Array of Frames of length at most 6, desired resolution res
    Output: One frame with ordered pictures side by side 
    """
    frames = _order_frames(enclosure_code, enclosure_video_code, frames)
    img_black = np.zeros([res[1],res[0],3],dtype=np.uint8)
    if amount_streams == 1:
        vis = frames[0]
            
    elif amount_streams == 2:            
        vis = np.concatenate((frames[0], frames[1]), axis=1)
            
    elif amount_streams == 3:
        vis1 = np.concatenate((frames[0], frames[1]), axis=1)            
        vis2 = np.concatenate((frames[2], img_black), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)

    elif amount_streams == 4:
        vis1 = np.concatenate((frames[0], frames[1]), axis=1)
        vis2 = np.concatenate((frames[2], frames[3]), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
            
    elif amount_streams == 5:
        vis1 = np.concatenate((frames[0], frames[1], frames[2]), axis=1)
        vis2 = np.concatenate((frames[3], frames[4], img_black), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)

    elif amount_streams == 6:
        vis1 = np.concatenate((frames[0], frames[1], frames[2]), axis=1)
        vis2 = np.concatenate((frames[3], frames[4], frames[5]), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)

    return vis


def _get_intervals_for_training_balanced(label_arr, label_dist, poss_behav = POSSIBLE_BEHAVIORS):
    
    ret = []   
    counter_dict = {}
    for i in label_dist.keys():
        counter_dict[i] = 0
    
    for interval_num in range(len(label_arr)):
    # check whether current interval should be taken into account
        curr_behav = label_arr[interval_num]

        # Skip out intervals
        if curr_behav == _get_out_code(poss_behav):
            continue
        
        if counter_dict[curr_behav] % label_dist[curr_behav] > 0:
            counter_dict[curr_behav] += 1
            continue
        ret.append(interval_num)
        counter_dict[curr_behav] += 1
    
    return ret
        

def _create_pictures_from_videos_with_labels_ssim_intervals(videolist, labelfile, enclosure_code, label_dist, interval_len = INTERVAL_LEN,
                                                            behav_map = BEHAVIOR_MAPPING, poss_behav = POSSIBLE_BEHAVIORS, 
                                                            multiple_frame_dir = OUTPUT_PATH_MULTIPLE_FRAME, delim_label = CSV_DELIMITER_LABELLING):
    """
    Input: videolist and corresponding labelfile
    Output: No ouput.
    I/O operations:
    - merges videos to one image
    """
    
    vid_nums = []
    for vid in videolist:
        vid_nums.append(vid.split('_')[-1].split('.')[0])                    
    vid_string = '+'.join(vid_nums)                
    enclosure_video_code = '{}*{}'.format(enclosure_code, vid_string)
    
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Creating pictures from videos with labelfile " + labelfile + " for enclosure " + enclosure_video_code + ".")
    print("The current frame modulus is given by " + str(label_dist))
    cut_off = get_cutoff(enclosure_code)
    label_arr = _get_labelling_sequence(csv_filename = labelfile, interval_len=interval_len, cut_off = cut_off,
                                        behav_map = behav_map, delim = delim_label, map_behavior = True, get_intervals = True)
    
    
    which_intervals = _get_intervals_for_training_balanced(label_arr, label_dist)
    
    

    
    videos = []
    width_dims = []
    height_dims = []
    for vid_path in videolist:
        vcap = cv2.VideoCapture(vid_path)
        videos.append(vcap)
        width_dims.append(int(vcap.get(3)))
        height_dims.append(int(vcap.get(4)))
    
    interval_num = 0
    frame_num = 0
    success = True

    if len(videolist) == 0:
         return

    date, individualname = _get_individualinfo_from_labelfile(labelfile)
    frames_per_interval = _get_frames_per_interval()

    frames = []

    while success:
        frames_suc = []
        for vid in videos:
            suc, frame = vid.read()
            frames_suc.append( (suc, frame) )
            success = success*suc

        if not success:
            continue

        frames = [x[1] for x in frames_suc]
        
        
        if interval_num == 0:
            success, res = _decide_width_height(width_dims, height_dims, len(videolist))
            
        if interval_num >= len(label_arr):
            success = False
            print("Finished after "+str(frame_num)+" frames.")
            
        
        if frame_num % interval_len == 0 and frame_num >= 1:
            interval_num += 1
        
        if not interval_num in which_intervals:
            frame_num += 1 
            continue
        

        if not success:
            frame_num += 1            
            continue
        
   
        # check whether current interval should be taken into account
        curr_behav = label_arr[interval_num]
        
        if frame_num % 5000 == 0 and interval_num > 0:
            percent = 100*np.round((1.0* interval_num / (1.0*len(label_arr))) , 2)
            print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ": Processing of "+str(frame_num)+" frames is done (" + str(percent) + "%).")
         
            

        if frame_num % interval_len not in frames_per_interval:
            frame_num += 1
            continue
        
                


        # rescale each frame
        for i in range(len(frames)):
            frames[i] = cv2.resize(frames[i], res, interpolation=cv2.INTER_AREA)

        
        # concatenate single pictures        
        vis = _concatenate_frames(enclosure_code, enclosure_video_code, frames, res, len(videolist))
       
        # turn whole image to grayscale        
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
        
        # add black polygones (TODO: maybe add the polygon array as parameter here?)        
        vis = _add_black_polygon(vis, enclosure_video_code)
        
        desig_path = multiple_frame_dir + enclosure_code + "/" + date + "/" + individualname + "/" + str(curr_behav) + "/" + str(interval_num) + "/"
        desig_filename = date + "_" + individualname + "_" + str(curr_behav) + "_" +str(frame_num).zfill(7) + ".jpg"
        _save_image(frame = vis, path = desig_path, filename = desig_filename)
            
        
        frame_num += 1

        


    print("**********************************************************************")






def _get_frames_per_interval(interval_len = INTERVAL_LEN, images_per_interval = IMAGES_PER_INTERVAL):
    return [int(1 + x * (interval_len - 1)/images_per_interval) for x in range(images_per_interval)]



    
def _get_date_from_videofile(videopath):
    """
    Input: path to videofile.
    Output: something like 2017-10-09_Elen_Kronberg_3.avi (where the 3 is the videonumber)
    requirement: Filename is like: /home/path/.../2017-10-09_Elen_Kronberg_3_SUM-10s_pred.csv
    """
    parts = videopath.split("/")[-1].split("_")
    return parts[0]

def _get_individualinfo_from_labelfile(labelfile):
    """
    Input: path to labelfile.
    Output: something like 2017-10-09, Elen_Kronberg_3 (where the 3 is the individual)
    requirement: Filename is like: /home/path/.../2017-10-09_Elen_Kronberg_3_SUM-10s_pred.csv
    """
    parts = labelfile.split("/")[-1].split("_")
    return parts[0], parts[1]+"_"+parts[2]+"_"+parts[3]


def _order_frames(enclosure_key, enclosure_video_code, frame_arr, configuration = BOX_PLACEMENT):
    """
    Input: array of frames from up to 6 videofiles.
    Output: the same frames but in an order that is given by the configuration
    """
    if not enclosure_key in configuration and not enclosure_video_code in configuration:
        return frame_arr
    
    if enclosure_video_code in configuration:
        perm = configuration[enclosure_video_code]
    elif enclosure_key in configuration:
        perm = configuration[enclosure_key]
    if len(perm) != len(frame_arr):
        return frame_arr
    
    s = sorted(perm)    
    perm_rel = [s.index(x) for x in perm]

    ret_list = [frame_arr[i] for i in perm_rel]

    return ret_list


def _add_black_polygon(img, config, polygon_mapping = POLYGON_ENDPOINTS):
    """
    Input: image (np array), enclosure_code and an information where to put black polygons
    Output: Returns original image in there is no information for the given enclosure,
    otherwise it will add the designated black polygons
    """
    
    enc_vid_code = config
    enc_code = enc_vid_code.split('*')[0]
    
    if enc_vid_code in polygon_mapping.keys():
        for polygon in polygon_mapping[enc_vid_code]:
            pts = polygon
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img, np.int32([pts]), (0,0,0))
        return img
    
    if enc_code in polygon_mapping.keys():
        for polygon in polygon_mapping[enc_code]:
            pts = polygon
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(img, np.int32([pts]), (0,0,0))
        return img
    
    return img

def generate_training_data(box_placement = BOX_PLACEMENT, polygon_endpoints = POLYGON_ENDPOINTS, all_behaviors = ALL_BEHAVIORS,
                           possible_behaviors = POSSIBLE_BEHAVIORS, weight_factor = WEIGHT_FACTOR, behavior_mapping = BEHAVIOR_MAPPING,
                           csv_overview = CSV_OVERVIEW_FILE, csv_overview_delim = CSV_DELIMITER_OVERVIEW, animal_num_sep = ANIMAL_NUM_SEPERATOR,
                           csv_label_delim = CSV_DELIMITER_LABELLING, interval_len = INTERVAL_LEN, base_data_path = BASE_PATH_TO_DATA,
                           multiple_frame_dir = OUTPUT_PATH_MULTIPLE_FRAME):

    # Preparation: Get labelling distribution and the frame-modulus array
    vid_label_map = _create_video_label_mapping(overview_file = csv_overview, delim = csv_overview_delim)
    label_dist_per_ind = _get_labelling_distribution_per_individual(video_label_map_dict = vid_label_map, all_behaviors = all_behaviors, csv_delim_labelling = csv_label_delim)
    real_label_dist_per_ind = _map_labellings_to_real_labels(individual_label_overview = label_dist_per_ind, behavior_map = behavior_mapping, all_behaviors = all_behaviors, possible_behaviors = possible_behaviors)
    
    frame_modulus_array = _get_labelling_dist(label_dict = real_label_dist_per_ind, possible_behaviors = possible_behaviors, weight_factor = weight_factor)

    # get images out of the video files
    for enclosure_code in vid_label_map:
        amount_individuals = len(vid_label_map[enclosure_code])
        print("Starting with enclosure "+enclosure_code)        
        print("Enclosure contains "+str(amount_individuals)+" individuals.")
        for individual_num in vid_label_map[enclosure_code]:
            print("Starting with data of individual "+individual_num+", containing "+str(len(vid_label_map[enclosure_code][individual_num][0]))+" nights.")
            videoliste = vid_label_map[enclosure_code][individual_num][0]
            csvliste = vid_label_map[enclosure_code][individual_num][1]
            label_distribution = frame_modulus_array[enclosure_code][individual_num]
            
            for j in range(len(videoliste)):
                videos = videoliste[j]
                labelfile = csvliste[j]                
                _create_pictures_from_videos_with_labels_ssim_intervals(videolist = videos, labelfile = labelfile, enclosure_code = enclosure_code,
                                                                 label_dist = label_distribution, interval_len = interval_len,
                                                                 behav_map = behavior_mapping, poss_behav = possible_behaviors,
                                                                 multiple_frame_dir = OUTPUT_PATH_MULTIPLE_FRAME, delim_label = CSV_DELIMITER_LABELLING)
    print("Finished processing.")
   


        
        
def predict_one_timerinterval(path_to_timeinterval, model, 
                              output_folder_mf, output_folder_sf,
                              interval_num, individual_code, possible_labels,
                              min_confidency = MIN_CONFIDENCY,
                              max_detections = MAX_DETECTIONS,
                              img_size = (300,300)):
    
    def _individual_name_from_boxcode(individual_codes, yolo_label):
        if len(individual_codes) == 1:
            return individual_codes[0]

        if yolo_label.startswith("Elenantilope"):
            yolo_label = yolo_label.replace("Elenantilope", "Elen")
        return yolo_label
    
    def _post_process_boxes(curr_detection, label_names):

        if len(curr_detection) > 1:
            label_box_list = sorted(curr_detection, key=itemgetter(5), reverse = True)
            if label_box_list[0][0] == label_box_list[1][0]: # both animals have the same label
                poss_names = [name for name in label_names]
                poss_names.remove(label_box_list[0][0])
                label_box_list[1][0] = poss_names[0]
                
                
            return label_box_list[0:2]
        
        return curr_detection
    
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    
    
    def ensure_new_folder_structure(output_base = YOLO_OUTPUT_FOLDER, classes = len(BEHAVIOR_NAMES_BASE)):
        for j in range(classes):
            ensure_dir(output_base + 'Single_Frame/Bilder/' + str(j))
            ensure_dir(output_base + 'Multiple_Frame/Bilder/' + str(j))
            
    if len( os.listdir(path_to_timeinterval) ) == 0:
        return
    
    images = []
    for img_name in sorted(os.listdir(path_to_timeinterval)):
        img_base = img_name[:-12]
        img_path = path_to_timeinterval + img_name
        
        img = cv2.imread(img_path)
        
        x = model.predict(img_path, plot_img = False)
        x = x.sort_values(by='score', ascending=False)

        label_box_list = []
        label_names = model.class_names
        for box_index in range(min(len(x),max_detections)):
            if x.iloc[box_index]['score'] >= min_confidency:    
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
                label_box_list_postprcessed = _post_process_boxes(curr_detection = label_box_list,
                                                                  label_names = label_names)
                
            if len(label_names) == 1:
                label_box_list_postprcessed = [sorted(label_box_list, key=itemgetter(5), reverse = True)[0]]
        
        if len(label_box_list) >= 1:
            
            if len(label_names) == 1:
                label_box = label_box_list_postprcessed[0]
                x1, y1 = label_box[1], label_box[2]
                x2, y2 = label_box[3], label_box[4]
                        
                box_part = img[y1:y2, x1:x2]
                images.append(box_part)
            if len(label_names) >= 1:
                for label, x1, y1, x2, y2, score in label_box_list_postprcessed:
                    if not label == individual_code:
                        continue
                    box_part = img[y1:y2, x1:x2]
                    images.append(box_part)
        
    if len(images) == 0:
        return
    
    single_frame_img = cv2.resize(images[0], img_size)
    multiple_frame_img = make_joint_image(images)
    
    interval = interval_num.zfill(7)
    img_path_save = img_base + '_' + interval + '.jpg' 
    
    ensure_dir(output_folder_sf)
    ensure_dir(output_folder_mf)
    
    cv2.imwrite(output_folder_sf + img_path_save, single_frame_img)
    cv2.imwrite(output_folder_mf + img_path_save, multiple_frame_img)
        

def  make_joint_image(frames, res = (150,150)):

    img_black = np.zeros([res[1],res[0],3],dtype=np.uint8)
    amount_streams = len(frames)
    frames = [cv2.resize(f, res) for f in frames ]
    
  
    if amount_streams == 1:
        vis1 = np.concatenate((frames[0], img_black), axis=1)            
        vis2 = np.concatenate((img_black, img_black), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
            
    elif amount_streams == 2:            
        vis1 = np.concatenate((frames[0], frames[1]), axis=1)            
        vis2 = np.concatenate((img_black, img_black), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
            
    elif amount_streams == 3:
        vis1 = np.concatenate((frames[0], frames[1]), axis=1)            
        vis2 = np.concatenate((frames[2], img_black), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)

    elif amount_streams == 4:
        vis1 = np.concatenate((frames[0], frames[1]), axis=1)
        vis2 = np.concatenate((frames[2], frames[3]), axis=1)
        vis = np.concatenate((vis1, vis2), axis=0)
            

    return vis

def predict_folder(input_base, output_base, model, individual_code):
    
    for behavior in sorted(os.listdir(input_base + enclosure_code + '/' + datum + '/' + individual_code)):
        print(enclosure_code + '/' + datum + '/' + individual_code + '/' + behavior)
        for time_interval in sorted(os.listdir(input_base + enclosure_code + '/' + datum + '/' + individual_code + '/' + behavior)):
            curr_path = input_base + enclosure_code + '/' + datum + '/' + individual_code + '/' + behavior + '/' + time_interval + '/'
                                   
            sf_out = output_base + 'Single_Frame/Bilder/' + behavior + '/'
            mf_out = output_base + 'Multiple_Frame/Bilder/' + behavior + '/'
            predict_one_timerinterval(path_to_timeinterval = curr_path, 
                                      model = model, 
                                      individual_code = individual_code,
                                      possible_labels = model.class_names,
                                      output_folder_mf = mf_out, 
                                      output_folder_sf = sf_out,
                                      interval_num = time_interval)
                        
if __name__ == "__main__":   
    if CREATE_IMAGES:
        generate_training_data()
    
    if CREATE_YOLO_CUTOUT:
        
        for enclosure_code in sorted(os.listdir(OUTPUT_PATH_MULTIPLE_FRAME)):
            
            net, label = get_yolo_net(enclosure_code)
            model = Yolov4(class_name_path=label)
            model.load_model(net)                         
            
            for datum in sorted(os.listdir(OUTPUT_PATH_MULTIPLE_FRAME + enclosure_code )):
                for individual_code in sorted(os.listdir(OUTPUT_PATH_MULTIPLE_FRAME + enclosure_code + '/' + datum)):
                    
                    predict_folder(input_base = OUTPUT_PATH_MULTIPLE_FRAME, 
                           output_base = YOLO_OUTPUT_FOLDER, 
                           model = model,
                           individual_code = individual_code)
            
        

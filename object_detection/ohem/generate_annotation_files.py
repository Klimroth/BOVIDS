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
from global_configuration import OD_NETWORK_LABELS_GLOBAL as GLOBAL_LABELS
from global_configuration import BASE_OD_NETWORK_GLOBAL as GLOBAL_NETS
from global_configuration import get_object_detection_network

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from models import Yolov4
from operator import itemgetter



##############################################################################

CREATE_IMAGES = True
CSV_OVERVIEW_FILE = '' # CSV File that contains the information which data should be used for training, or prepared for prediction.
BASE_PATH_TO_DATA = '' # Contains the starting point of navigation to the videos 
IMAGES_OUTPUT_FOLDER = ''

VIDEO_LEN_SPECIAL = {}
IMAGES_PER_INDIVIDUAL = 300

##############################################################################

CREATE_YOLO_ANNOTATION = False
YOLO_OUTPUT_FOLDER = ''
YOLO_NETPATH = ''

MAX_DETECTIONS = 1
MIN_CONFIDENCY = 0.9
   
############################################################################## 
    
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

VIDEO_HOURS = 14
VIDEO_LEN = VIDEO_HOURS * 3600
CUT_OFF_SECONDS = VIDEO_LEN + 1


def get_yolo_net(enclosure_code, 
                 glob_nets = GLOBAL_NETS, 
                 glob_labels = GLOBAL_LABELS, 
                 path = YOLO_NETPATH):
    
    net, label = get_object_detection_network(enclosure_code = enclosure_code, 
                    enclosure_individual_code = 'not implemented',
                     basenets = glob_nets, 
                     labels = glob_labels)
    
    
    return path + net, path + label
    

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
    

def _create_videolist_for_prediction(overview_file = CSV_OVERVIEW_FILE, delim = CSV_DELIMITER_OVERVIEW):
    """
    Returns a dictionary {Art_Zoo_Enclosure_Num: {date: video_list_per_day} }
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
                    return_dict[dict_key] = {date: avi_video_filelist }
                else:
                    return_dict[dict_key][date] = avi_video_filelist

            line_count += 1

    return return_dict







def _save_image(frame, path, filename):
    """
    Saves the np.array() frame as an image to path. 
    """
    if not os.path.exists(path):
        os.makedirs(path)


    
    x = cv2.imwrite(path + filename, frame) 
    
    if not x:
        print("Error", path + filename)

    

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


    



def _get_date_from_videofile(videopath):
    """
    Input: path to videofile.
    Output: something like 2017-10-09_Elen_Kronberg_3.avi (where the 3 is the videonumber)
    requirement: Filename is like: /home/path/.../2017-10-09_Elen_Kronberg_3_SUM-10s_pred.csv
    """
    parts = videopath.split("/")[-1].split("_")
    return parts[0]


def _order_frames(enclosure_key, enclosure_video_code, frame_arr, configuration = BOX_PLACEMENT):
    """
    Input: array of frames from up to 6 videofiles.
    Output: the same frames but in an order that is given by the configuration
    """
    
    print("Enclosure:", enclosure_key)
    print("Enclosure Video:", enclosure_video_code)
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


def _generate_pictures_for_labelling_one_date(amount_pictures, 
                                              vid_len, 
                                              videolist, 
                                              output_path, 
                                              enclosure_code, 
                                              enclosure_video_code,
                                              fps = 1, cut_off = CUT_OFF_SECONDS):
    
    print(enclosure_video_code)
    print(videolist)
    print("Amount images: "+str(amount_pictures))
    frame_mod = int(1.0*VIDEO_LEN*vid_len*fps/(14*amount_pictures))

    videos = []
    width_dims = []
    height_dims = []
    for vid_path in videolist:
        vcap = cv2.VideoCapture(vid_path)
        videos.append(vcap)
        width_dims.append(int(vcap.get(3)))
        height_dims.append(int(vcap.get(4)))

    frame_num = 0
    success = True

    if len(videolist) == 0:
         return
    
    date, individualname = videolist[0].split("/")[-1].split("_")[0],  enclosure_code.split("_")[0] + '_' + enclosure_code.split("_")[1] + '_' + enclosure_code.split("_")[2].split(".")[0]

    while success:        
        
        frames_suc = []
        for vid in videos:
            suc, frame = vid.read()
            frames_suc.append( (suc, frame) )
            success = success*suc
        
        if frame_num > cut_off:
            success = False
            
        if not success:
            continue
        
        frames = [x[1] for x in frames_suc]

        if frame_num == 0:
            success, res = _decide_width_height(width_dims, height_dims, len(videolist))
        
        if frame_num % 5000 == 0:
            print("Processed frames: "+str(frame_num))
            
        if frame_num % frame_mod > 0:
            frame_num += 1
            continue
            
        if not success:
            frame_num += 1            
            continue
        
        
            

        frame_num += 1

        # rescale each frame
        for i in range(len(frames)):
            frames[i] = cv2.resize(frames[i], res, interpolation=cv2.INTER_AREA)


        # concatenate single pictures
        vis = _concatenate_frames(enclosure_code, enclosure_video_code, frames, res, len(videolist))
       
        # turn whole image to grayscale
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
        
        # add black polygones (TODO: maybe add the polygon array as parameter here?)
        vis = _add_black_polygon(vis, enclosure_video_code)

        # save the whole frame
                
        desig_path = output_path
        desig_filename = date + "_" + individualname + "_" +str(frame_num).zfill(7) + ".jpg"
        _save_image(frame = vis, path = desig_path, filename = desig_filename)
        
    

    
    

def _prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'latin1')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def _create_xml_file(file_name, file_path, img_size, 
                     label_box_list, output_file):

    '''

    Parameters
    ----------
    containing_folder : string
        Path to the folder the image file belongs to.
    file_name : string
        The image name (without .jpg).
    file_path : string
        Complete path to the image file.
    img_size : TYPE
        DESCRIPTION.
    label_box_list : list
        [ [label, xmin, ymin, xmax, ymax], ... ].
    output_folder : TYPE, optional
        DESCRIPTION. The default is OUTPUT_FOLDER_LABELS.

    Returns
    -------
    None.

    '''
    # create tree itself
    annotation = Element('annotation')

    #folder = SubElement(annotation, 'folder')
    #folder.text = containing_folder
    
    filename = SubElement(annotation, 'filename')
    filename.text = file_name
    
    path = SubElement(annotation, 'path')
    path.text = file_path
    
    src = SubElement(annotation, 'source')
    db = SubElement(src, 'database')
    db.text = 'Unknown'
    
    segmented = SubElement(annotation, 'segmented')
    segmented.text = "0"
   
    # put in image size and depth (=1)
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_size[0])
    height = SubElement(size, 'height')
    height.text = str(img_size[1])
    depth = SubElement(size, 'depth')
    depth.text = str(img_size[2])
    
    for label_box in label_box_list:
        label = str(label_box[0])
        xmin_, ymin_ = str(label_box[1]), str(label_box[2])
        xmax_, ymax_ = str(label_box[3]), str(label_box[4])
                
        # create object
        obj = SubElement(annotation, 'object')
        
        # put in label
        name = SubElement(obj, 'name')
        name.text = label
        
        pose = SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        
        truncated = SubElement(obj, 'truncated')
        truncated.text = "0"
        
        difficult = SubElement(obj, 'difficult')
        difficult.text = "0"

        # put in box coordinates
        bndbox = SubElement(obj, 'bndbox')
        
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = xmin_
        
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = ymin_
        
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = xmax_
        
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = ymax_
    
    annotation_pretty =  _prettify(annotation) 
    
    output_path = os.path.dirname(output_file) + '/'
    ensure_dir(output_path)
    with open(output_file, "w+", encoding = 'latin1') as f:
        f.write(annotation_pretty)



def create_label_files(model, img_path_list, mode,
                       in_path_base = IMAGES_OUTPUT_FOLDER,
                       out_path_base = YOLO_OUTPUT_FOLDER,
                       max_detections = MAX_DETECTIONS,
                       min_confidency = MIN_CONFIDENCY,
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
    

    label_names = model.class_names
    j = 1
    for img_path in img_path_list:    
        
        img = cv2.imread(img_path)
        h,w,c = img.shape
        
        x = model.predict(img_path, plot_img = False)
        x = x.sort_values(by='score', ascending=False)

        label_box_list = []
        
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
            
            
                
            
            file_name = img_path.split("/")[-1][:-4]
            output_file = img_path.replace(in_path_base, out_path_base)[:-4] + '.xml'
            _create_xml_file(file_name = file_name, 
                                 file_path = img_path,
                                 img_size = (h,w,c), 
                                 label_box_list = label_box_list_postprcessed, 
                                 output_file = output_file)
        
                
        else:
            print("No bounding box for image " + str(img_path))

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_files_and_folders(root, extension = '.jpg'):
    #This will print a list of all the directories in your tree first and the print the paths of all the files in the directory recursively.
    #C:\hello\my_folder
    #C:\hello\another_folder
    #C:\hello\my_folder\abc.txt
    #C:\hello\my_folder\xyz.txt
    #C:\hello\another_folder\new.txt
    
    filelist = []
    folderlist = []
    for root, dirs, files in os.walk(root):
        for d in dirs:
            folderlist.append(os.path.join(root, d))    
        for f in files:
            if f.endswith(extension):
                filelist.append(os.path.join(root, f))
    return filelist, folderlist



def ensure_new_folder_structure(folder_names, input_base = IMAGES_OUTPUT_FOLDER,
                                output_base = YOLO_OUTPUT_FOLDER):
    for f in folder_names:
        f_new = f.replace(input_base, output_base)
        ensure_dir(f_new)
    
    

def create_annotations(model, input_folder, mode = 'create_annotations'):
    files, folders = get_files_and_folders(input_folder, extension = '.jpg')
    ensure_new_folder_structure(folders)
    create_label_files(model=model, img_path_list=files, mode = mode)
    






        
if __name__ == "__main__":   
    
    overview = _create_videolist_for_prediction()
    if CREATE_IMAGES:
        ensure_dir(IMAGES_OUTPUT_FOLDER)
        for enclosure_code in overview.keys():
            for date in overview[enclosure_code]:
                vid_list = overview[enclosure_code][date]
                pics_per_night = int(IMAGES_PER_INDIVIDUAL / len(overview[enclosure_code]))
                art, zoo, enc = enclosure_code.split('_')
                output_folder = '{}{}/{}/{}/'.format(IMAGES_OUTPUT_FOLDER, art, zoo, enc)                
                
                vid_nums = []
                for vid in vid_list:
                    vid_nums.append(vid.split('_')[-1].split('.')[0])                    
                vid_string = '+'.join(vid_nums)                
                enclosure_video_code = '{}*{}'.format(enclosure_code, vid_string)
                
                vid_len = 14
                if zoo in VIDEO_LEN_SPECIAL.keys():
                    vid_len = VIDEO_LEN_SPECIAL[zoo]
                _generate_pictures_for_labelling_one_date(amount_pictures = pics_per_night, vid_len = vid_len, 
                                                              fps = 1, videolist = vid_list,
                                                output_path = output_folder, enclosure_code = enclosure_code,
                                                enclosure_video_code = enclosure_video_code)
    
    if CREATE_YOLO_ANNOTATION:
        print("Create Annotations.")
        for species in sorted(os.listdir(IMAGES_OUTPUT_FOLDER)):
            for zoo in sorted(os.listdir(IMAGES_OUTPUT_FOLDER + species)):
                for enclosure in sorted(os.listdir(IMAGES_OUTPUT_FOLDER + species + '/' + zoo)):
                    curr_path = IMAGES_OUTPUT_FOLDER + species + '/' + zoo + '/' + enclosure + '/'
                    enclosure_code = species + '_' + zoo + '_' + enclosure
                    print("***", enclosure_code)
                    
                    net, label = get_yolo_net(enclosure_code)
                    model = Yolov4(class_name_path=label)
                    model.load_model(net)       
                           
                    create_annotations(model, input_folder = curr_path)
        
        

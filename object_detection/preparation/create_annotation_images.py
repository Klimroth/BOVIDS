#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

import numpy as np
import os
import cv2



BASE_PATH_VIDEO = ''
OUTPUT_FOLDER_BASE = ''

ZOO = ''
ENCLOSURE_NUMBER = '' # enclosure number as a string
SPECIES = ''
VIDEO_NUMBERS = [4]
LIST_OF_DATES = ['']
VID_LEN_HOURS = 14



NUMBER_IMAGES_CREATE = 500 


BOX_PLACEMENT = {
                 
}

POLYGON_ENDPOINTS ={
      
        }









"""
****************************************************************************
"""

VIDEO_LEN = int(50400*VID_LEN_HOURS/14)
BASE_KAESTCHEN = '{}{}/{}/Videos/'.format(BASE_PATH_VIDEO, SPECIES, ZOO)
VIDEO_LIST_KAESTCHEN = [ '{}_{}'.format(SPECIES, vidnum) for vidnum in VIDEO_NUMBERS] # Videonummer
ZOONAME_KAESTCHEN = ZOO
CONFIGURATION_NAME_KAESTCHEN = VIDEO_LIST_KAESTCHEN[0].split("_")[0] + "_" + ZOONAME_KAESTCHEN + "_" + ENCLOSURE_NUMBER
OUTPUT_FOLDER_KAESTCHEN = OUTPUT_FOLDER_BASE + VIDEO_LIST_KAESTCHEN[0].split("_")[0] + "/" + ZOONAME_KAESTCHEN + "/" + ENCLOSURE_NUMBER + "/"





def _save_image(frame, path, filename):
    """
    Saves the np.array() frame as an image to path. 
    """
    if not os.path.exists(path):
        os.makedirs(path)

    script_path = os.getcwd()
    os.chdir(path)
    x = cv2.imwrite(filename, frame)
    os.chdir(script_path)

     
    
    if not x:
        print("Error: can not save file.", path + filename)

    

def  _concatenate_frames(enclosure_code, frames, res, amount_streams):
    """
    Input: Enclosure_code, Array of Frames of length at most 6, desired resolution res
    Output: One frame with ordered pictures side by side 
    """
    frames = _order_frames(enclosure_code, frames)
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


def _order_frames(enclosure_key, frame_arr, configuration = BOX_PLACEMENT):
    """
    Input: array of frames from up to 6 videofiles.
    Output: the same frames but in an order that is given by the configuration
    """
    if not enclosure_key in configuration:
        return frame_arr

    perm = configuration[enclosure_key]

    if len(perm) != len(frame_arr):
        return frame_arr
    
    ret_list = [frame_arr[i-1] for i in perm]
    return ret_list

def _add_black_polygon(img, config, polygon_mapping = POLYGON_ENDPOINTS):
    """
    Input: image (np array), enclosure_code and an information where to put black polygons
    Output: Returns original image in there is no information for the given enclosure,
    otherwise it will add the designated black polygons
    """
    if not config in polygon_mapping:
        return img
    
    for polygon in polygon_mapping[config]:
        pts = polygon
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, np.int32([pts]), (0,0,0))
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


def _generate_pictures_for_labelling_one_date(amount_pictures = 500, vid_len = 14, fps = 1, videolist = VIDEO_LIST_KAESTCHEN,
                                    output_path = OUTPUT_FOLDER_KAESTCHEN, enclosure_code = CONFIGURATION_NAME_KAESTCHEN):
    
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
        vis = _concatenate_frames(enclosure_code, frames, res, len(videolist))
       
        # turn whole image to grayscale
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
        
        # add black polygones (TODO: maybe add the polygon array as parameter here?)
        vis = _add_black_polygon(vis, enclosure_code)

        # save the whole frame
                
        desig_path = output_path
        desig_filename = date + "_" + individualname + "_" +str(frame_num).zfill(7) + ".jpg"
        _save_image(frame = vis, path = desig_path, filename = desig_filename)
        
    
    
def generate_pictures_for_labelling(amount_pictures = NUMBER_IMAGES_CREATE, vid_len = 14, fps = 1, basis_folder = BASE_KAESTCHEN, videolist = VIDEO_LIST_KAESTCHEN, datum_liste = LIST_OF_DATES,
                                    output_path = OUTPUT_FOLDER_KAESTCHEN, enclosure_code = CONFIGURATION_NAME_KAESTCHEN, zooname = ZOONAME_KAESTCHEN):
    
    videoliste_complete = []
    err = False
    
    for date in datum_liste:
        vid_list_date = []
        for video_nummer in videolist:
            animal_name = video_nummer.split("_")[0]
            video_num = video_nummer.split("_")[1]
            vid_path = basis_folder+video_nummer+"/"+date+"_"+animal_name+"_"+zooname+"_"+video_num+".avi"
            if os.path.isfile(vid_path):
                vid_list_date.append(basis_folder+video_nummer+"/"+date+"_"+animal_name+"_"+zooname+"_"+video_num+".avi")
            else:
                err = True
                print("ERROR: "+vid_path+" does not exist.")
        videoliste_complete.append(vid_list_date)
    
    if err:
        return
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    pics_per_night = int(amount_pictures / len(videoliste_complete)) + 1
    
    for video_liste_per_night in videoliste_complete:
        _generate_pictures_for_labelling_one_date(amount_pictures = pics_per_night, vid_len = vid_len, fps = fps, videolist = video_liste_per_night,
                                    output_path = output_path, enclosure_code = enclosure_code)
            
    
    
    
    
    

generate_pictures_for_labelling()
        
        
        
        

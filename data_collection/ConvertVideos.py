# -*- coding: cp1252 -*-
"""ConvertVideos_v1.py: Converts single .asf-Files from observation systems into .avi files containing videos of one night / one day."""
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2019"
__credits__ = ["Jennifer Gübert"]
__license__ = "GPL-3.0"
__version__ = "1.1"
__status__ = "Development"

import numpy as np
import cv2
import os

###############################################################
############# Parameters ######################################
###############################################################

# Relative path to Videos
PATH_TO_VID = 'Z:/Promotion Jenny/Münster/Gnu/Box_1/1/1'
CONVERT_WHOLE_FOLDER = False # (True = "neu", False = "alt")
# Set it to true in order to convert the typical structure as follows:
# Datapath: PATH_TO_VID contains Box_1, ..., Box_n
# Outputpath: OUTPUT_PATH + ANIMAL_NAME + Box-Number

ALREADY_SORTED = False # Is there already a folder structure?
# '.' if file is contained in the same folder as the videos
# e.g. 'folder1/folder2/folder3'

# Filename-Parameters 
ANIMAL_NAME = 'Wildebeest'
ZOO_NAME = 'Fantasyzoo'
OUTPUT_PATH = 'V:/Wildebeest/Fantasyzoo/Videos'
# Enter complete path.
# '.' for current folder
# e.g. 'V:/folder1/folder2/folder3'

# Time depending parameters
NIGHT_TIME_BEGIN = '17'
DAY_TIME_BEGIN = '07'

# Only configure if CONVERT_WHOLE_FOLDER = False: Sets the Videonumber as name
VIDEO_NUMBER = '1'




# Video Configuration
ORIGINAL_RES = True #keep original resolution?
MAXIMAL_WIDTH = 1280 # rescale to width (720p = 1280)
ORIGINAL_FPS = False #keep original fps?
OUTPUT_FPS = 1 # How many FPS in the output?

# Other parameters
KI_EXPORT = False # Export a grayscale version as well?
TEST_PROC = False # Is it just a test run? (Take 10sec from every file)



def _get_ASF_from_folder(path_folder):
    files = [f for f in os.listdir(path_folder) if os.path.isfile(path_folder+"/"+f) and f.endswith(".asf")]
    files.sort()
    return files

def _get_ASF_per_night(path_folder):
    ret_list = []
    
    asf_files = _get_ASF_from_folder(path_folder)
    for j in range(len(asf_files)):
        curr_asf_name = asf_files[j]
        curr_asf_cont = curr_asf_name.split("_")
        curr_begin_date = curr_asf_cont[3]
        curr_year = curr_begin_date[0:4]
        curr_month = curr_begin_date[4:6]
        curr_day = curr_begin_date[6:8]
        curr_start_time = curr_begin_date[8:10]
        
       
        if curr_start_time == NIGHT_TIME_BEGIN:
            # create new night
            ret_list.append([])
            curr_list = ret_list[-1]
            curr_list.append(curr_asf_name)
        elif curr_start_time == DAY_TIME_BEGIN:
            # create new day
            ret_list.append([])
            curr_list = ret_list[-1]
            curr_list.append(curr_asf_name)
        else:
            if len(ret_list) == 0:
                ret_list.append([])
            curr_list = ret_list[-1]
            curr_list.append(curr_asf_name)

    return ret_list
    
def _get_filename(asf_night_list, animal, zoo, num):
    curr_asf_name = asf_night_list[0]
    curr_asf_cont = curr_asf_name.split("_")
    curr_begin_date = curr_asf_cont[3]
    curr_year = curr_begin_date[0:4]
    curr_month = curr_begin_date[4:6]
    curr_day = curr_begin_date[6:8]
    curr_start_time = curr_begin_date[8:10]

    ending = "_Undefined"
    if curr_start_time == DAY_TIME_BEGIN:
        ending = "_Tag"
    if curr_start_time == NIGHT_TIME_BEGIN:
        ending = ""

    ret_name = curr_year + "-" + curr_month + "-" + curr_day + "_" + animal + "_" + zoo + "_" + num + ending

    return ret_name


def _createFolders(ki_export = KI_EXPORT, path = OUTPUT_PATH):
    if not os.path.exists(path+"/"):
        os.makedirs(path+"/")
    if ki_export:
        if not os.path.exists(path+"/original"):
            os.makedirs(path+"/original")
    
def _get_subfolders(path_to_data):
    filenames = [f for f in os.listdir(path_to_data) if os.path.isdir(path_to_data+"/"+f)]
    return filenames


def _convertVideosOfFolder(animal, zoo, num, already_sorted, KI, path_to_data, path_to_save, test_proc, max_res, orig_res, orig_fps, out_fps):        
    if already_sorted:
        subfolders = _get_subfolders(path_to_data)
    else:
        subfolders = ['.']

    _createFolders(ki_export = KI, path = path_to_save)
    for directory in subfolders:
        asf_nights = _get_ASF_per_night(path_to_data+"/"+directory)
        for asf_night in asf_nights:
            output_name = _get_filename(asf_night, animal, zoo, num)
            _CreateVideoFromFiles(output_name = output_name, listOfFiles = asf_night, sub_path = path_to_data+"/"+directory,
                                  test = test_proc, maxRes = max_res, KI = KI, original_res = orig_res, original_fps = orig_fps,
                                  output_fps = out_fps, path_to_save = path_to_save)
            
            


def convertVideos(animal = ANIMAL_NAME, zoo = ZOO_NAME, num = VIDEO_NUMBER, already_sorted = ALREADY_SORTED,
                  KI = KI_EXPORT, path_to_data = PATH_TO_VID, path_to_save = OUTPUT_PATH, convert_folder = CONVERT_WHOLE_FOLDER,
                  test = TEST_PROC, original_res = ORIGINAL_RES, output_fps = OUTPUT_FPS, max_width = MAXIMAL_WIDTH, orig_fps = ORIGINAL_FPS):

    if path_to_save.endswith('/'):
        path_to_save = path_to_save[:-1]
    if path_to_data.endswith('/'):
        path_to_data = path_to_data[:-1]

    if convert_folder:
        for box_name in os.listdir(path_to_data):
            if not box_name.startswith("Box_"):
                print(box_name+" has the wrong format.")
                continue
                
            video_num = box_name.split("_")[1]
            output_path = path_to_save + '/' + animal + '_' + video_num
            data_path = path_to_data + '/' + box_name
            print("###################################################")
            print("Converting Videos from folder: "+data_path+" into outputfolder "+output_path)
            
            _convertVideosOfFolder(animal = animal, zoo = zoo, num = video_num, already_sorted = already_sorted,
                                   KI = KI, path_to_data = data_path, path_to_save = output_path, test_proc = test,
                                   max_res = max_width, orig_res = original_res, orig_fps = orig_fps, out_fps = output_fps)
    else:
        print("###################################################")
        print("Converting Videos from folder: "+ path_to_data +" into outputfolder "+path_to_save)
        _convertVideosOfFolder(animal = animal, zoo = zoo, num = num, already_sorted = already_sorted,
                                   KI = KI, path_to_data = path_to_data, path_to_save = path_to_save, test_proc = test,
                                   max_res = max_width, orig_res = original_res, orig_fps = orig_fps, out_fps = output_fps)
                
            
def _CreateVideoFromFiles(output_name, listOfFiles, sub_path, test, maxRes, KI, original_res, original_fps, output_fps, path_to_save):
    print("**********************************")
    print("Beginning with file " + sub_path + "/" + output_name)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    

    # get resolution
    first_file = listOfFiles[0]
    cap1 = cv2.VideoCapture(sub_path+"/"+first_file)    

    rescale = False
    if not original_res:
        if int(cap1.get(3)) > maxRes:
            downscale_fac_orig = maxRes/cap1.get(3)
            res = (maxRes, int(downscale_fac_orig*cap1.get(4)))
            rescale = True
        else:
            res = (int(cap1.get(3)), int(cap1.get(4)))
    else:
        res = (int(cap1.get(3)), int(cap1.get(4)))

    KI_width = min(int(cap1.get(3)), 1280)
    downscale_fac = KI_width/cap1.get(3)
    downscale_h = downscale_fac*cap1.get(4)
    res2 = (KI_width, int(downscale_h))

    # get fps
    input_fps = cap1.get(cv2.CAP_PROP_FPS)
    if original_fps:
        output_fps = input_fps
    fps_ratio = int(input_fps/output_fps)
    if fps_ratio < 1:
        fps_ratio = 1
    print("FPS ratio between input file ("+str(input_fps)+") and output file ("+str(output_fps)+") is: "+str(fps_ratio))
    cap1.release()
    
    fps = 1
    
    if KI:
        out = cv2.VideoWriter(path_to_save+"/original/"+output_name+".avi",fourcc, output_fps, res)
        out2 = cv2.VideoWriter(path_to_save+"/"+output_name+".avi",fourcc, output_fps, res2, isColor=False)
    else:
        out = cv2.VideoWriter(path_to_save+"/"+output_name+".avi",fourcc, output_fps, res)

    for sFile in listOfFiles:
        print("Processing " + sFile)
        cap = cv2.VideoCapture(sub_path+"/"+sFile)
        ret = True
        if test: # Testing procedure, just get 90sec from each video!
            iFrame = 1
            while(ret and iFrame < 11*input_fps):
                iFrame += 1
                ret, frame = cap.read()
                if ret == False:
                    continue
                if iFrame % fps_ratio == 0:
                    if KI:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.resize(frame_gray, res2)
                    if rescale:
                        frame = cv2.resize(frame, res)
                    if ret==True:
                        out.write(frame)
                        if KI:
                            out2.write(frame_gray)
                    else:
                        break
        else:
            iFrame = 0
            while(ret):
                iFrame += 1
                ret, frame = cap.read()
                if ret == False:
                    continue
                if iFrame % fps_ratio == 0:
                    if KI:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.resize(frame_gray, res2)
                    if rescale:
                        frame = cv2.resize(frame, res)
                    if ret==True:
                        out.write(frame)
                        if KI:
                            out2.write(frame_gray)
                    else:
                        break
        cap.release()
        print(sFile+" consists of "+str(iFrame)+" frames.")
                
    out.release()
    if KI:
        out2.release()
    cv2.destroyAllWindows()

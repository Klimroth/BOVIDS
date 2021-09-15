__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

import cv2
import numpy as np
from datetime import datetime


OUTPUT_DESTINATION = 'U:/output.avi'
OUTPUT_FPS = 1
VIDEOLISTE = {
    'U:/stream_1.avi' : 1,
    'U:/stream_2.avi' : 4,
    'U:/stream_3.avi' : 2,
    'U:/stream_4.avi' : 3
    }


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = [x for x in dictOfElements.items()]

    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys[0]

def _order_frames(frame_dict, ordering = VIDEOLISTE):
    ret = []
    tmp = []
    for x in sorted(ordering.values()):
        y = getKeysByValue(ordering, x)
        tmp.append(y)
    for x in tmp:
        ret.append(frame_dict[x])
    return ret

def  _concatenate_frames(frames, res, amount_streams):
    """
    Input: Enclosure_code, Array of Frames of length at most 6, desired resolution res
    Output: One frame with ordered pictures side by side 
    """
    frames = _order_frames(frames)
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

def _decide_width_height(width_dims, height_dims, amount_streams):
    is_low_res = False
    wi = [x for x in width_dims.values()]
    hi = [x for x in height_dims.values()]

    overall = (0,0)
    
    ratios = [wi[i]*1.0/hi[i] for i in range(len(wi))]
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
            overall = (800, 600)
        else:
            res = (1280, 720)
            overall = (1280, 720)
    elif amount_streams == 2:        
        if is_low_res:
            res = (640, 480)
            overall = (1280, 480)
        else:
            res = (640, 360)
            overall = (1280, 360)
    elif amount_streams in [3,4]:        
        if is_low_res:
            res = (640, 480)
            overall = (1280, 960)
        else:
            res = (640, 360)
            overall = (1280, 720)
    elif amount_streams in [5,6]:
        if is_low_res:
            res = (426, 320)
            overall = (1278, 960)
        else:
            res = (426, 240)
            overall = (1278, 720)
    else:
        print("Error: It is currently not supported to have more than 6 video streams!")
        return False, res, overall
    return True, res, overall

def merge_videos( videolist = VIDEOLISTE):
    success = True
    
    videos = {}
    width_dims = {}
    height_dims = {}
    
    frame_num = 0
    for vid_path in videolist.keys():
        if not vid_path.endswith(".avi"):
            exit("Error: Files need to finish with .avi")
        vcap = cv2.VideoCapture(vid_path)
        videos[vid_path] = vcap
        width_dims[vid_path] = int(vcap.get(3))
        height_dims[vid_path] = int(vcap.get(4))
    while success:
        frames_succesful = {}
        at_least_one_suc = False
        j = 0
        for vid_path in videos.keys():
            vid = videos[vid_path]
            suc, frame = vid.read()
            if suc:
               frames_succesful[vid_path]  = frame
               at_least_one_suc = True
            else:
                img_black = np.zeros([width_dims[vid_path], height_dims[vid_path], 3],dtype=np.uint8)
                frames_succesful[vid_path]  = img_black

        success = at_least_one_suc
        if not success:
            continue
        
        if frame_num == 0:
            success, res, overall_res = _decide_width_height(width_dims, height_dims, len(videolist.keys()))
            fourcc = cv2.VideoWriter_fourcc(*'FMP4')
            output_vid = cv2.VideoWriter(OUTPUT_DESTINATION,fourcc, OUTPUT_FPS, overall_res)
            print(res, overall_res)
            
        if frame_num % 1000 == 0:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("Processed "+str(frame_num)+" many frames.")
        frame_num += 1

        if not success:
            continue

        for j in frames_succesful.keys():
            frames_succesful[j] = cv2.resize(frames_succesful[j], res)
        vis = _concatenate_frames(frames_succesful, res, len(videolist.keys()))
        vis = cv2.cvtColor(vis,cv2.COLOR_RGB2BGR)
     
        output_vid.write(vis)

    output_vid.release()
    for vid in videos.values():
        vid.release()    
    cv2.destroyAllWindows() 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2020"
__credits__ = ["Jennifer Gübert"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

"""
1) Convert into 1 fps: convert_fps(original = 25, new = 1, savepath = '')
2) Concatenate video files:  merge_videos()
"""

from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2, os
import numpy as np



"""

Insert black frames
- fill_in_frames() 
"""

# fill_in_frames()
VIDEO_PATH = "V:/Oryx/FantasyZoo/Videos/Oryx_2/" # containing folder
VID_NAME = "2020-03-09_Oryx_Fantasyzoo_2.avi" # video filename
FILL_FRAMES = [["09:00:1", "02:00:04", "03:00:00"]]
# [minute in videofile, time (day) before, time (day) after]

"""
Reduce FPS
- convert_fps()
"""
VIDEO_BASE_PATH = "V:/Oryx/FantasyZoo/Videos/Oryx_2/" # containing folder
VIDEO_NUMBER = 'Oryx_Fantasyzoo_3' # soecies_zoo_videonumber
DATES = ['2018-01-14', '2018-01-15']
EXT = '' #
FPS_NEW = 1
SAVE_FOLDER = 'D:/somewhere/'


"""
Concatenate videos
- merge_videos() ausführen
"""

PART1 = 'V:/vid1.avi'
PART2 = 'V:/vid2.avi'
SAVEPATH = 'V:/merged_video.avi'
WIDTH = 1280 # desired width of output.

"""
*************************************************************
"""





def fill_in_frames(vid_path = VIDEO_PATH, vid_name = VID_NAME):
    
    def get_fill_positions(pos_arr = FILL_FRAMES):

        def convert_string(timestring):
            h,m,s = map(int, timestring.split(":"))
            return h,m,s
    
        fill_pos = []
        fill_len = []
    
        shift = 0
        for pos in pos_arr:
            h,m,s = convert_string(pos[0])
            vid_pos = s+60*m+60*60*h
            fill_pos.append(vid_pos + shift)
    
            h,m,s = convert_string(pos[1])
            h2,m2,s2 = convert_string(pos[2])
            
            if h >= 12 and h2 <= 12: # DIRTY, JUST WORKS IN MOST CASES
                h2 += 24
                
            length = s2+60*m2+60*60*h2 - (s+60*m+60*60*h)
            fill_len.append(length)
    
            shift += length
            
        
        return fill_pos, fill_len

    vid = cv2.VideoCapture(vid_path + vid_name)
    suc = True
    
    vidname = vid_name.split(".")[0]

    # Video params
    res = (int(vid.get(3)), int(vid.get(4)))
    input_fps = 1
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    out = cv2.VideoWriter(vid_path+vidname+"_new.avi",fourcc, input_fps, res)
    black_frame = np.zeros((res[1], res[0], 3))

    frame_num = 0
    curr_fill_pos = 0

    

    fill_positions, fill_lengths = get_fill_positions()
    print("Positions to fill:" + str(fill_positions))
    print("Amount of black frames: " + str(fill_lengths))
    while suc:
        suc, frame = vid.read()

        if frame_num % 5000 == 0:
            print("Processed frames: "+str(frame_num))
        
        if suc:

            if curr_fill_pos < len(fill_positions):
                if fill_positions[curr_fill_pos] == frame_num:
                    for j in range(fill_lengths[curr_fill_pos]):
                        out.write(black_frame)
                        frame_num += 1
                    curr_fill_pos += 1
            out.write(frame)

        frame_num += 1        
        
    vid.release()
    out.release()

def merge_videos(p1 = PART1, p2 = PART2, save = SAVEPATH):
    
    clip_1 = VideoFileClip(p1).resize(width=WIDTH)
    clip_2 = VideoFileClip(p2).resize(width=WIDTH)
    final_clip = concatenate_videoclips([clip_1,clip_2], method="compose")
    final_clip.write_videofile(filename = save, codec='mpeg4', audio=False, fps=1)
    
    clip_1.close()
    clip_2.close()
    final_clip.close()

def convert_fps():
    for date in sorted(DATES):
        
        path = VIDEO_BASE_PATH + date + '_' + VIDEO_NUMBER + EXT + '.avi'
        path_new = SAVE_FOLDER + date + '_' + VIDEO_NUMBER + '.avi'
        print(path)
        
        clip = VideoFileClip(path)
        new_clip = clip.set_fps(FPS_NEW)  
        new_clip.write_videofile(path_new, codec = 'mpeg4', fps=FPS_NEW)
        clip.close()
        new_clip.close()
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.2"
__status__ = "Development"


"""
Contains all the needed configurations, i.e.

# Object Detection
- BASE_OD_NETWORK = { 'species': path to od-network.h5 } (for those enclosures using the basenet)
- ZOO_OD_NETWORK = { 'species_zoo' : path to od-network.h5 } (for those enclosures using a different net)
- ENCLOSURE_OD_NETWORK = { 'species_zoo_enclosure' : path to od-network.h5 } (for those enclosures using a different net)
- OD_NETWORK_LABELS = {'key': [list of labels]}
- MINIMUM_CONFIDENCY


# Creating images from videos
- VIDEO_ORDER_PLACEMENT = { 'species_zoo_enclosure': [2,1] } # only gets an entry if the order of the videos should not be the natural order
- VIDEO_BLACK_REGIONS = { 'species_zoo_enclosure': [ np.array() ]  } # contains the polygon endpoints of the black areas in the videos
- CSV_DELIMITER
- ANIMAL_NUMBER_SEPERATOR


"""

from global_configuration import *

"""
Saving paths
"""

INPUT_CSV_FILE = '' 

# on local hard drive
TMP_STORAGE_IMAGES = ''
TMP_STORAGE_CUTOUT = '' 

# on server
FINAL_STORAGE_CUTOUT = ''
FINAL_STORAGE_PREDICTION_FILES = ''

LOGGING_FILES = ''



"""
Conducted Steps (False = step will be conducted)
"""
SKIP_IMAGE_CREATION = False
                             
SKIP_INDIVIDUAL_DETECTION = False

SKIP_BEHAVIOR_TOTAL_SF = False
SKIP_BEHAVIOR_TOTAL_MF = False

SKIP_BEHAVIOR_BINARY_SF = False
SKIP_BEHAVIOR_BINARY_MF = False  
                             
SKIP_MOVING_FILES = False
SKIP_REMOVING_TEMPORARY_FILES = False
                             
SKIP_PP_TOTAL = False
SKIP_PP_BINARY = False

SKIP_OD_DENSITY = False

"""
General data paths
"""
BASE_PATH_DATA = ''
BEHAVIOR_NETWORK_BASEPATH = ''
OD_NETWORK_BASEPATH = ''

""" 
General configuration
"""
STANDARD = True # if True, outputs one normal file and one binary prediction.
# if false, one might vary the following.
BEHAVIOR_MAPPING = {0:0, 1:1, 2:2, 3:3, 4:4}


CUT_OFF = 7200



"""
Behavior Prediction
"""

BEHAVIOR_NETWORK_JOINT = {}
BEHAVIOR_NETWORK_SINGLE_FRAME = {}

BEHAVIOR_NETWORK_JOINT_BINARY = {}
BEHAVIOR_NETWORK_SINGLE_FRAME_BINARY = {}

for k, v in BEHAVIOR_NETWORK_JOINT_GLOBAL.items():
    BEHAVIOR_NETWORK_JOINT[k] = BEHAVIOR_NETWORK_BASEPATH + v 

for k, v in BEHAVIOR_NETWORK_SINGLE_FRAME_GLOBAL.items():
    BEHAVIOR_NETWORK_SINGLE_FRAME[k] = BEHAVIOR_NETWORK_BASEPATH + v 
    
for k, v in BEHAVIOR_NETWORK_JOINT_GLOBAL_BINARY.items():
    BEHAVIOR_NETWORK_JOINT_BINARY[k] = BEHAVIOR_NETWORK_BASEPATH + v 

for k, v in BEHAVIOR_NETWORK_SINGLE_FRAME_GLOBAL_BINARY.items():
    BEHAVIOR_NETWORK_SINGLE_FRAME_BINARY[k] = BEHAVIOR_NETWORK_BASEPATH + v 


"""
IMAGE CUTOUT
"""
BASE_OD_NETWORK = {}
OD_NETWORK_LABELS = {}

for k, v in BASE_OD_NETWORK_GLOBAL.items():
    BASE_OD_NETWORK[k] = OD_NETWORK_BASEPATH + v

for k, v in OD_NETWORK_LABELS_GLOBAL.items():
    OD_NETWORK_LABELS[k] = OD_NETWORK_BASEPATH + v


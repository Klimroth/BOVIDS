#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.1"
__maintainer__ = "M. Hahn-Klimroth"
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

import numpy as np
from post_processing_rules import INTERVAL_LENGTH, POST_PROCESSING_RULES, OUT_REGULATIONS


"""
General
"""

BEHAVIOR_NAMES_BASE = ['standing', 'lying', 'sleeping']
BEHAVIOR_NAMES_SPECIAL = ['out', 'truncated']
BEHAVIOR_NAMES = BEHAVIOR_NAMES_BASE + BEHAVIOR_NAMES_SPECIAL

COLOR_MAPPING = {'standing': 'cornflowerblue', 
                 'lying': 'forestgreen', 
                 'sleeping': 'limegreen', 
                 'out': 'darkgrey',
                 'truncated': 'grey'}


IMG_SIZE = (300, 300)
BATCH_SIZE_BEHAVIOR = 8

GLOBAL_STARTING_TIME = 17
GLOBAL_ENDING_TIME = 7
GLOBAL_OBSERVATION_HOURS = 14 
GLOBAL_VIDEO_LENGTH = GLOBAL_OBSERVATION_HOURS*3600

PACKAGE_SIZE_DENSITY_STATISTICS = 0.5 # hours






"""
POST-PROCESSING
"""

PP_RULE_DICT = {
    # species: [total, binary]
    'Addax': ['Standard', 'Standard-binary'],
    
}



"""
IMAGE CUTOUT
"""
DETECTION_MIN_CONFIDENCE = 0.90

# species
# enclosure_code
# enclosure (1) individual_code (2 und 3): species_zoo_1_2+3
MIN_DETECTION_SCORE = {
        'Addax': 0.2,

}


IOU_THRESHOLD = {
        }

"""
IMAGE CREATION FROM VIDEOS
"""
CSV_DELIMITER = ","
ANIMAL_NUMBER_SEPERATOR = ";"
IMAGES_PER_INTERVAL = 4

VIDEO_ORDER_PLACEMENT = {
                
}

VIDEO_BLACK_REGIONS ={
      
        }



"""
Detection Networks
"""
STANDARD_OD = ''

BASE_OD_NETWORK_GLOBAL = {
    # species
    'Addax': STANDARD_OD,
        
    # species_zoo
    
    # species_zoo_enclosure
     
      
    # species_zoo_enclosure_individuals e.g. 'Eland_Place_1_1+2'
}

STANDARD_OD_LABEL = ''

OD_NETWORK_LABELS_GLOBAL = {
    'Addax': STANDARD_OD_LABEL,
    # species_zoo
    
    # species_zoo_enclosure
    
    # 'Bergriedbock_Kronberg_1': 'zusammenstehende/2022-01-13_OD_Bergriedbock_Kronberg_1/classes.txt',
 
    # species_zoo_enclosure_individuals e.g. 'Elen_Hannover_1_1+2'

    }



"""
******************************************************************************
Action classification
******************************************************************************
"""
# standard-variablen

BEHAVIOR_NETWORK_JOINT_GLOBAL = {
    
    'Addax': 'network_file.h5',
 

    }

BEHAVIOR_NETWORK_SINGLE_FRAME_GLOBAL = {
    'Addax': 'network_file.h5',
  }

BEHAVIOR_NETWORK_JOINT_GLOBAL_BINARY = {
    'Addax': 'network_file.h5',
    }

BEHAVIOR_NETWORK_SINGLE_FRAME_GLOBAL_BINARY = {
 'Addax': 'network_file.h5',
    }




TRUNCATION_TOP_STANDARD = 0
TRUNCATION_BOT_STANDARD = 2000
TRUNCATION_LEFT_STANDARD = 0
TRUNCATION_RIGHT_STANDARD = 2000

def get_iou_thresh(enclosure_code_ind_code):
    
    enclosure_code = enclosure_code_ind_code.split('_')[0] + '_' + enclosure_code_ind_code.split('_')[1] + '_' + enclosure_code_ind_code.split('_')[2]
    if enclosure_code_ind_code in IOU_THRESHOLD.keys():
        return IOU_THRESHOLD[enclosure_code_ind_code]
    if enclosure_code in IOU_THRESHOLD.keys():
        return IOU_THRESHOLD[enclosure_code]
    return 0.5

def get_detection_score(enclosure_code_ind_code):
    enclosure_code = enclosure_code_ind_code.split('_')[0] + '_' + enclosure_code_ind_code.split('_')[1] + '_' + enclosure_code_ind_code.split('_')[2]
    species, _, _ = enclosure_code.split('_')
    
    if enclosure_code_ind_code in MIN_DETECTION_SCORE.keys():
        return MIN_DETECTION_SCORE[enclosure_code_ind_code]
    
    if enclosure_code in MIN_DETECTION_SCORE.keys():
        return MIN_DETECTION_SCORE[enclosure_code]
    
    if species in MIN_DETECTION_SCORE.keys():
        return MIN_DETECTION_SCORE[species]
    
    return DETECTION_MIN_CONFIDENCE

def get_behaviornetwork(individual_code, network_dict):
    
    if individual_code in network_dict.keys():
        print("Used AC network: ", network_dict[individual_code])
        return network_dict[individual_code]
        
    species, zoo = individual_code.split("_")[0], individual_code.split("_")[1]
        
    if species + '_' + zoo in network_dict.keys():
        print("Used AC network: ", network_dict[species + '_' + zoo])
        return network_dict[species + '_' + zoo]
        
    if species in network_dict.keys():
        print("Used AC network: ", network_dict[species])
        return network_dict[species]
    
    print("ERROR: No suitable AC network found in global_configuration!")
    return ''

def get_object_detection_network(enclosure_code, 
                    enclosure_individual_code,
                     basenets, 
                     labels):
    
    species =  enclosure_code.split('_')[0]
    zoo_code = enclosure_code.split('_')[0] + enclosure_code.split('_')[1]
    
    net = False
    label = False
    
    
    if enclosure_individual_code in basenets.keys():
        net = basenets[enclosure_individual_code]
        label = labels[enclosure_individual_code]
    elif enclosure_code in basenets.keys():
        net = basenets[enclosure_code]
        label = labels[enclosure_code]
    elif zoo_code in basenets.keys():
        net = basenets[zoo_code]
        label = labels[zoo_code]
    elif species in basenets.keys():
        net = basenets[species]
        label = labels[species]
    
        
    return net, label


def get_postprocessing_rule(individual_code, mode, pp_set = PP_RULE_DICT):
    species, zoo, indnum = individual_code.split('_')
    if individual_code in pp_set.keys():
        return pp_set[individual_code][mode] # mode = 0 -> total, 1 -> binary
    if species in pp_set.keys():
        return pp_set[species][mode]
    
    print('ERROR: No post-processing rule defined for ', individual_code)
    return ''

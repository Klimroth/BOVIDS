#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.1"
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


INTERVAL_LENGTH = 7
IMG_SIZE = (300, 300)
BATCH_SIZE_BEHAVIOR = 8

GLOBAL_STARTING_TIME = 17
GLOBAL_ENDING_TIME = 7
GLOBAL_OBSERVATION_HOURS = 14
GLOBAL_VIDEO_LENGTH = GLOBAL_OBSERVATION_HOURS*3600

PACKAGE_SIZE_DENSITY_STATISTICS = 0.5 # hours

"""
Post-Processing
"""
POST_PROCESSING_RULES = {
'no': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 0,
       'ROLLING_AVERAGE_JOINT_IMAGES': 0,
       'ROLLING_AVERAGE_ENSEMBLE': 0,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5],
       
        'MIN_LEN_SLS': 0,
        'MIN_LEN_SLA': 0,
        'MIN_LEN_ALS': 0,
        'MIN_LEN_ALA': 0,
        'MIN_LEN_OLA': 0,
        'MIN_LEN_OLS': 0,
        'MIN_LEN_ALO': 0,
        'MIN_LEN_SLO': 0,
        
        'MIN_LEN_SAS': 0,
        'MIN_LEN_SAL': 0,
        'MIN_LEN_LAS': 0,
        'MIN_LEN_LAL': 0,
        'MIN_LEN_LAO': 0,
        'MIN_LEN_OAL': 0,
        'MIN_LEN_OAS': 0,
        'MIN_LEN_SAO': 0,
        
        'MIN_LEN_ASA': 0,
        'MIN_LEN_ASL': 0,
        'MIN_LEN_LSA': 0,
        'MIN_LEN_LSL': 0,
        'MIN_LEN_LSO': 0,
        'MIN_LEN_OSL': 0,
        'MIN_LEN_ASO': 0,
        'MIN_LEN_OSA': 0,
        
        'MIN_LEN_OUT': 0,
        'MIN_TIME_OUT': 0,
        
        'MIN_LEN_TRUNCATION': 0, # shorter truncations will just count as the previous behavior.
        'MIN_LEN_TRUNCATION_SWAP': 7200, # longer truncations will count as REAL_BEHAVIOR_LONG
        'TRUNCATION_REAL_BEHAVIOR_LONG': 1, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 3,
        
        'OUT_FLUCTUATION_REMOVAL_MAX': 3*9,
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20 },
        
'Standard': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 4*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':4,
       'ROLLING_AVERAGE_ENSEMBLE': 4,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5],
       
        'MIN_LEN_SLS': 3,
        'MIN_LEN_SLA': 3,
        'MIN_LEN_ALS': 3,
        'MIN_LEN_ALA': 6,
        'MIN_LEN_OLA': 6,
        'MIN_LEN_OLS': 6,
        'MIN_LEN_ALO': 6,
        'MIN_LEN_SLO': 6,
        
        'MIN_LEN_SAS': 25,
        'MIN_LEN_SAL': 25,
        'MIN_LEN_LAS': 25,
        'MIN_LEN_LAL': 25,
        'MIN_LEN_LAO': 25,
        'MIN_LEN_OAL': 25,
        'MIN_LEN_OAS': 25,
        'MIN_LEN_SAO': 25,
        
        'MIN_LEN_ASA': 9,
        'MIN_LEN_ASL': 6,
        'MIN_LEN_LSA': 6,
        'MIN_LEN_LSL': 2,
        'MIN_LEN_LSO': 9,
        'MIN_LEN_OSL': 9,
        'MIN_LEN_ASO': 9,
        'MIN_LEN_OSA': 9,
        
        'MIN_LEN_OUT': 9,
        'MIN_TIME_OUT': 0,
        
        'MIN_LEN_TRUNCATION': 13, # shorter truncations will just count as the previous behavior.
        'MIN_LEN_TRUNCATION_SWAP': 85, # longer truncations will count as REAL_BEHAVIOR_LONG
        'TRUNCATION_REAL_BEHAVIOR_LONG': 1, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 3,
        
        'OUT_FLUCTUATION_REMOVAL_MAX': 3*9,
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20 },

'Standard-binary': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 5*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':5,
       'ROLLING_AVERAGE_ENSEMBLE': 5,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5],
        
        'MIN_LEN_SLS': 0,
        'MIN_LEN_SLA': 0,
        'MIN_LEN_ALS': 0,
        'MIN_LEN_ALA': 5*9,
        'MIN_LEN_OLA': 5*9,
        'MIN_LEN_OLS': 0,
        'MIN_LEN_ALO': 5*9,
        'MIN_LEN_SLO': 0,
        
        'MIN_LEN_SAS': 0,
        'MIN_LEN_SAL': 0,
        'MIN_LEN_LAS': 0,
        'MIN_LEN_LAL': 5*9,
        'MIN_LEN_LAO': 5*9,
        'MIN_LEN_OAL': 5*9,
        'MIN_LEN_OAS': 0,
        'MIN_LEN_SAO': 0,
        
        'MIN_LEN_ASA': 0,
        'MIN_LEN_ASL': 0,
        'MIN_LEN_LSA': 0,
        'MIN_LEN_LSL': 0,
        'MIN_LEN_LSO': 0,
        'MIN_LEN_OSL': 0,
        'MIN_LEN_ASO': 0,
        'MIN_LEN_OSA': 0,       
      
        
        'MIN_LEN_OUT': 5*9,
        'MIN_TIME_OUT': 0,
        
        'MIN_LEN_TRUNCATION': 43, # shorter truncations will just count as the previous behavior.
        'MIN_LEN_TRUNCATION_SWAP': 85, # longer truncations will count as REAL_BEHAVIOR_LONG
        'TRUNCATION_REAL_BEHAVIOR_LONG': 1, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 3,
        
        'OUT_FLUCTUATION_REMOVAL_MAX': 15*9,
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20},

}



"""
IMAGE CUTOUT
"""
DETECTION_MIN_CONFIDENCE = 0.90

MIN_DETECTION_SCORE = {
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

BASE_OD_NETWORK_GLOBAL = {
    # species
        
    # species_zoo
    
    # species_zoo_enclosure

    # species_zoo_enclosure_individuals e.g. 'Eland_ZooName_1_1+2'
}


OD_NETWORK_LABELS_GLOBAL = {
 
    # species
       
    # species_zoo
    
    # species_zoo_enclosure
 
    # species_zoo_enclosure_individuals e.g. 'Eland_ZooName_1_1+2'
        
    }

BEHAVIOR_NETWORK_JOINT_GLOBAL = {
    
    # species_zoo
    
    # individualcode

    }

BEHAVIOR_NETWORK_SINGLE_FRAME_GLOBAL = {
    # species_zoo
    
    # individualcode
    }


BEHAVIOR_NETWORK_JOINT_GLOBAL_BINARY = {
    # species_zoo
    
    # individualcode
    }

BEHAVIOR_NETWORK_SINGLE_FRAME_GLOBAL_BINARY = {
    # species_zoo
    
    # individualcode
    }




def get_iou_thresh(enclosure_code_ind_code):
    
    enclosure_code = enclosure_code_ind_code.split('_')[0] + '_' + enclosure_code_ind_code.split('_')[1] + '_' + enclosure_code_ind_code.split('_')[2]
    if enclosure_code_ind_code in IOU_THRESHOLD.keys():
        return IOU_THRESHOLD[enclosure_code_ind_code]
    if enclosure_code in IOU_THRESHOLD.keys():
        return IOU_THRESHOLD[enclosure_code]
    return 0.5

def get_detection_score(enclosure_code_ind_code):
    enclosure_code = enclosure_code_ind_code.split('_')[0] + '_' + enclosure_code_ind_code.split('_')[1] + '_' + enclosure_code_ind_code.split('_')[2]
    
    if enclosure_code_ind_code in MIN_DETECTION_SCORE.keys():
        return MIN_DETECTION_SCORE[enclosure_code_ind_code]
    if enclosure_code in MIN_DETECTION_SCORE.keys():
        return MIN_DETECTION_SCORE[enclosure_code]
    return DETECTION_MIN_CONFIDENCE

def get_behaviornetwork(individual_code, network_dict):
    
    if individual_code in network_dict.keys():
        return network_dict[individual_code]
        
    species, zoo = individual_code.split("_")[0], individual_code.split("_")[1]
        
    if species + '_' + zoo in network_dict.keys():
        return network_dict[species + '_' + zoo]
        
    if species.startswith('Zebra'): # TODO: Add missing networks from time to time
        return network_dict['Basisnetzwerk_Zebra']
        
    return network_dict['Basisnetzwerk_Antilopen']

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

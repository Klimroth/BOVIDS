#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "M. Hahn-Klimroth"
__status__ = "Development"


import numpy as np


"""
Post-Processing
"""

POST_PROCESSING_MAPPING_TOTAL = {
    'Elen': 'Standard',
    'Elen_Dortmund_3': 'Elen-juv',
    'Elen_Dortmund_4': 'Elen-juv',
    'Elen_Kronberg_4': 'Elen-juv',
    'Elen_Kronberg_5': 'Elen-juv',
    'Elen_Münster_4': 'Elen-juv',
    'Elen_Münster_5': 'Elen-juv',
    'Elen_Münster_6': 'Elen-juv',
    'Elen_Münster_7': 'Elen-juv',
    'Blessbock': 'Bless',
    'Bongo': 'Bongo',
    'Gnu': 'Gnu',
    'Kudu': 'Kudu',
    'Okapi': 'Standard',
    'Pferdeantilope': 'Standard',
    'Rappenantilope': 'Bongo',
    'Sitatunga': 'Standard',
    'Zebra-Berg': 'Gnu',
    'Zebra-Grevy': 'Gnu',
    'Zebra-Steppen': 'Gnu'
    }

POST_PROCESSING_MAPPING_BINARY = {
    'Elen': 'Standard-binary', 
    'Blessbock': 'Standard-binary',
    'Bongo': 'Standard-binary',
    'Gnu': 'Standard-binary',
    'Kudu': 'Standard-binary',
    'Okapi': 'Standard-binary',
    'Pferdeantilope': 'Standard-binary',
    'Rappenantilope': 'Standard-binary',
    'Sitatunga': 'Standard-binary',
    'Zebra-Berg': 'Standard-binary',
    'Zebra-Grevy': 'Standard-binary',
    'Zebra-Steppen': 'Standard-binary'   
    }

"""
Time
"""
VIDEO_START_SPECIAL = {
    'Addax_Hannover': 18,
    'Blessbock_Gelsenkirchen': 19,
    'Blessbock_Hannover': 18,
    'Elen_Hannover': 18,
    'Kudu_Gelsenkirchen': 19,
    'Pferdeantilope_Hannover': 18,
    'Rappenantilope_Gelsenkirchen': 19,
    'Zebra-Steppen_Gelsenkirchen': 19,
    'Zebra-Steppen_Duisburg': 18
    
    }

VIDEO_END_SPECIAL = {
    'Säbelantilope_Leipzig': 6
    }








"""
General
"""

ERROR_TOLERANCE_INTERVALS = 4


BEHAVIOR_NAMES = ['Stehen', 'Liegen', 'Schlafen', 'Out', 'Truncation']
COLOR_MAPPING = { BEHAVIOR_NAMES[0]: 'cornflowerblue', 
                 BEHAVIOR_NAMES[1]: 'forestgreen', 
                 BEHAVIOR_NAMES[2]: 'limegreen', 
                 BEHAVIOR_NAMES[3]: 'darkgrey',
                 BEHAVIOR_NAMES[4]: 'grey'}

# Spalten der Confusion-Matrix
CONF_SPALTEN = ['Stehen', 'Liegen', 'Schlafen', 'Out', 'Truncation']
CONF_ZEILEN = ['Stehen', 'Liegen', 'Schlafen', 'Out']


INTERVAL_LENGTH = 7

GLOBAL_STARTING_TIME = 17
GLOBAL_ENDING_TIME = 7
GLOBAL_OBSERVATION_HOURS = 14
GLOBAL_VIDEO_LENGTH = GLOBAL_OBSERVATION_HOURS*3600






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

'Elen-juv': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 3*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':3,
       'ROLLING_AVERAGE_ENSEMBLE': 3,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5],
       
        'MIN_LEN_SLS': 2,
        'MIN_LEN_SLA': 3,
        'MIN_LEN_ALS': 3,
        'MIN_LEN_ALA': 6,
        'MIN_LEN_OLA': 6,
        'MIN_LEN_OLS': 6,
        'MIN_LEN_ALO': 6,
        'MIN_LEN_SLO': 6,
        
        'MIN_LEN_SAS': 6,
        'MIN_LEN_SAL': 6,
        'MIN_LEN_LAS': 6,
        'MIN_LEN_LAL': 6,
        'MIN_LEN_LAO': 6,
        'MIN_LEN_OAL': 6,
        'MIN_LEN_OAS': 6,
        'MIN_LEN_SAO': 6,
        
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
        'TRUNCATION_REAL_BEHAVIOR_LONG': 3, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 3,
        'OUT_FLUCTUATION_REMOVAL_MAX': 3*9,
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20 },

'Zebra': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 2*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES': 2,
       'ROLLING_AVERAGE_ENSEMBLE': 2,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [1.0, 0.0],
       
        'MIN_LEN_SLS': 1,
        'MIN_LEN_SLA': 3,
        'MIN_LEN_ALS': 3,
        'MIN_LEN_ALA': 6,
        'MIN_LEN_OLA': 6,
        'MIN_LEN_OLS': 6,
        'MIN_LEN_ALO': 6,
        'MIN_LEN_SLO': 6,
        
        'MIN_LEN_SAS': 6,
        'MIN_LEN_SAL': 6,
        'MIN_LEN_LAS': 6,
        'MIN_LEN_LAL': 6,
        'MIN_LEN_LAO': 6,
        'MIN_LEN_OAL': 6,
        'MIN_LEN_OAS': 6,
        'MIN_LEN_SAO': 6,
        
        'MIN_LEN_ASA': 9,
        'MIN_LEN_ASL': 6,
        'MIN_LEN_LSA': 6,
        'MIN_LEN_LSL': 1,
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
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20},

'Zebra-binary': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 5*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':5,
       'ROLLING_AVERAGE_ENSEMBLE': 5,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [1.0, 0.0],
        
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

'Kudu': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 2*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':2,
       'ROLLING_AVERAGE_ENSEMBLE': 2,
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
        'MIN_LEN_OLS': 3,
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
        'MIN_LEN_LSL': 1,
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
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20},

'Bless': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 2*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES': 2,
       'ROLLING_AVERAGE_ENSEMBLE': 2,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5],
       
        'MIN_LEN_SLS': 1,
        'MIN_LEN_SLA': 2,
        'MIN_LEN_ALS': 3,
        'MIN_LEN_ALA': 6,
        'MIN_LEN_OLA': 6,
        'MIN_LEN_OLS': 6,
        'MIN_LEN_ALO': 6,
        'MIN_LEN_SLO': 6,
        
        'MIN_LEN_SAS': 9,
        'MIN_LEN_SAL': 9,
        'MIN_LEN_LAS': 9,
        'MIN_LEN_LAL': 9,
        'MIN_LEN_LAO': 9,
        'MIN_LEN_OAL': 9,
        'MIN_LEN_OAS': 9,
        'MIN_LEN_SAO': 9,
        
        'MIN_LEN_ASA': 9,
        'MIN_LEN_ASL': 6,
        'MIN_LEN_LSA': 6,
        'MIN_LEN_LSL': 1,
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
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20},

'Gnu': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 2*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES': 2,
       'ROLLING_AVERAGE_ENSEMBLE': 2,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-15), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5],
       
        'MIN_LEN_SLS': 1,
        'MIN_LEN_SLA': 3,
        'MIN_LEN_ALS': 3,
        'MIN_LEN_ALA': 6,
        'MIN_LEN_OLA': 6,
        'MIN_LEN_OLS': 6,
        'MIN_LEN_ALO': 6,
        'MIN_LEN_SLO': 6,
        
        'MIN_LEN_SAS': 6,
        'MIN_LEN_SAL': 6,
        'MIN_LEN_LAS': 6,
        'MIN_LEN_LAL': 6,
        'MIN_LEN_LAO': 6,
        'MIN_LEN_OAL': 6,
        'MIN_LEN_OAS': 6,
        'MIN_LEN_SAO': 6,
        
        'MIN_LEN_ASA': 9,
        'MIN_LEN_ASL': 6,
        'MIN_LEN_LSA': 6,
        'MIN_LEN_LSL': 1,
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
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20},

'Bongo': {
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
        
        'MIN_LEN_SAS': 9,
        'MIN_LEN_SAL': 9,
        'MIN_LEN_LAS': 9,
        'MIN_LEN_LAL': 9,
        'MIN_LEN_LAO': 9,
        'MIN_LEN_OAL': 9,
        'MIN_LEN_OAS': 9,
        'MIN_LEN_SAO': 9,
        
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
        'OUT_FLUCTUATION_REMOVAL_MIN_BEHAV': 0.20 }
}

def get_vid_start_end(individual_code, 
                      vid_start_normal = GLOBAL_STARTING_TIME,
                      vid_end_normal = GLOBAL_ENDING_TIME,
                      vid_start_special = VIDEO_START_SPECIAL,
                      vid_end_special = VIDEO_END_SPECIAL):
    species, zoo, individual = individual_code.split('_')
    specieszoo = species + '_' + zoo
    
    start = vid_start_normal
    end = vid_end_normal
    
    if individual_code in vid_start_special.keys():
        start = vid_start_special[individual_code]
    elif specieszoo in vid_start_special.keys():
        start = vid_start_special[specieszoo]
    elif zoo in vid_start_special.keys():
        start = vid_start_special[zoo]
    
    
    if individual_code in vid_end_special.keys():
        end = vid_end_special[individual_code]
    elif specieszoo in vid_end_special.keys():
        end = vid_end_special[specieszoo]
    elif zoo in vid_end_special.keys():
        end = vid_end_special[zoo]
        
    return start, end

def get_pp_rule(individual_code, mode, pp_required, total = POST_PROCESSING_MAPPING_TOTAL,
                binary = POST_PROCESSING_MAPPING_BINARY):
    species, zoo, individual = individual_code.split('_')
    
    if not pp_required:
        return POST_PROCESSING_RULES['no']

    if mode == 'total':
        if individual_code in total.keys():
            return POST_PROCESSING_RULES[total[individual_code]]
        if species in total.keys():
            return POST_PROCESSING_RULES[total[species]]
        else:
            print('WARNING: No post-processing rule. Taking Standard.')
            return POST_PROCESSING_RULES['Standard']
    if mode == 'binary':
        if individual_code in binary.keys():
            return POST_PROCESSING_RULES[binary[individual_code]]
        if species in binary.keys():
            return POST_PROCESSING_RULES[binary[species]]
        else:
            print('WARNING: No post-processing rule. Taking Standard-binary.')
            return POST_PROCESSING_RULES['Standard-binary']
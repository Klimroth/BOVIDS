#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "M. Hahn-Klimroth"
__status__ = "Development"

import numpy as np

INTERVAL_LENGTH = 7


"""
OUT-Regeln
Individuencode: [ [liste mit Daten], Startintervalle  ]
"""
OUT_REGULATIONS ={
                    
 }

"""
Post-Processing
"""
POST_PROCESSING_RULES = {
'no': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 0,
       'ROLLING_AVERAGE_JOINT_IMAGES': 0,
       'ROLLING_AVERAGE_ENSEMBLE': 0,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-4), 1.0]),
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
        
        'MIN_LEN_OAO': 0,
        'MIN_LEN_OLO': 0,
        'MIN_LEN_OSO': 0,
        
        'MIN_LEN_OUT': 0,
        'MIN_TIME_OUT': 0,
        
        'MIN_LEN_TRUNCATION': 0, # shorter truncations will just count as the previous behavior.
        'MIN_LEN_TRUNCATION_SWAP': 7200, # longer truncations will count as REAL_BEHAVIOR_LONG
        'TRUNCATION_REAL_BEHAVIOR_LONG': 4, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 4,
        
        'OUT_FLUCTUATION_REMOVAL_MAX': 0,
        'OUT_FLUCTUATION_REMOVAL_BEHAV_MAX': 0,
        'OUT_FLUCTUATION_REMOVAL_PERC': 0.0,
        
        'OUT_FLUCTUATION_REMOVAL_MAX_A': 0,
        'OUT_FLUCTUATION_REMOVAL_BEHAV_MAX_A': 0,
        'OUT_FLUCTUATION_REMOVAL_PERC_A': 0.0},
        

'Standard': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 4*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':4,
       'ROLLING_AVERAGE_ENSEMBLE': 4,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-4), 1.0]),
       'ROLLING_AVERAGE_POTENCY_SF': 1.0,
       'ROLLING_AVERAGE_POTENCY_MF': 1.0,
       'ROLLING_AVERAGE_POTENCY_ENSEMBLE': 1.0,
       'WEIGHTS_NETWORKS': [0.5, 0.5], # [MF, SF]
       
        'MIN_LEN_SLS': 6,
        'MIN_LEN_SLA': 6,
        'MIN_LEN_ALS': 6,
        'MIN_LEN_ALA': 6,
        'MIN_LEN_OLA': 6,
        'MIN_LEN_OLS': 6,
        'MIN_LEN_ALO': 6,
        'MIN_LEN_SLO': 6,
        
        'MIN_LEN_SAS': 13,
        'MIN_LEN_SAL': 13,
        'MIN_LEN_LAS': 13,
        'MIN_LEN_LAL': 13,
        'MIN_LEN_LAO': 13,
        'MIN_LEN_OAL': 13,
        'MIN_LEN_OAS': 13,
        'MIN_LEN_SAO': 13,
        
        'MIN_LEN_ASA': 43,
        'MIN_LEN_ASL': 6,
        'MIN_LEN_LSA': 6,
        'MIN_LEN_LSL': 6,
        'MIN_LEN_LSO': 6,
        'MIN_LEN_OSL': 6,
        'MIN_LEN_ASO': 6,
        'MIN_LEN_OSA': 6,
        
        'MIN_LEN_OAO': 3*9,
        'MIN_LEN_OLO': 3*9,
        'MIN_LEN_OSO': 3*9,
        
        'MIN_LEN_OUT': 3*9,
        'MIN_TIME_OUT': 0,
        
        'MIN_LEN_TRUNCATION': 13, # shorter truncations will just count as the previous behavior.
        'MIN_LEN_TRUNCATION_SWAP': 85, # longer truncations will count as REAL_BEHAVIOR_LONG
        'TRUNCATION_REAL_BEHAVIOR_LONG': 3, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 3,
        
        'OUT_FLUCTUATION_REMOVAL_MAX': 15*9,
        'OUT_FLUCTUATION_REMOVAL_BEHAV_MAX': 10*9,
        'OUT_FLUCTUATION_REMOVAL_PERC': 0.33,
        
        'OUT_FLUCTUATION_REMOVAL_MAX_A': 30*9,
        'OUT_FLUCTUATION_REMOVAL_BEHAV_MAX_A': 20*9,
        'OUT_FLUCTUATION_REMOVAL_PERC_A': 0.4},

'Standard-binary': {
       'ROLLING_AVERAGE_SINGLE_FRAMES': 5*INTERVAL_LENGTH,
       'ROLLING_AVERAGE_JOINT_IMAGES':5,
       'ROLLING_AVERAGE_ENSEMBLE': 5,
       'ROLLING_AVERAGE_WEIGHTS': np.array([1.0, 1.0, 1.0, 10**(-4), 1.0]),
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
      
        
        'MIN_LEN_OAO': 7*9,
        'MIN_LEN_OLO': 7*9,
        'MIN_LEN_OSO': 0,
        
        'MIN_LEN_OUT': 7*9,
        'MIN_TIME_OUT': 0,
        
        'MIN_LEN_TRUNCATION': 43, # shorter truncations will just count as the previous behavior.
        'MIN_LEN_TRUNCATION_SWAP': 85, # longer truncations will count as REAL_BEHAVIOR_LONG
        'TRUNCATION_REAL_BEHAVIOR_LONG': 3, # 1:transfers those truncation of at least 70 seconds to lying, 4: stays truncated
        'TRUNCATION_INTERMEDIATE': 3,
        
        'OUT_FLUCTUATION_REMOVAL_MAX': 30*9,
        'OUT_FLUCTUATION_REMOVAL_BEHAV_MAX': 20*9,
        'OUT_FLUCTUATION_REMOVAL_PERC': 0.4,
        
        'OUT_FLUCTUATION_REMOVAL_MAX_A': 30*9,
        'OUT_FLUCTUATION_REMOVAL_BEHAV_MAX_A': 20*9,
        'OUT_FLUCTUATION_REMOVAL_PERC_A': 0.5},


}

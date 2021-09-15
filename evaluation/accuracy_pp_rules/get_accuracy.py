#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "M. Hahn-Klimroth"
__status__ = "Development"


from configuration import BEHAVIOR_NAMES as BEHAVIORS
from configuration import INTERVAL_LENGTH

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import os




def get_dates_standard(spec, zoo, ind, csv_folder):
    
    def _get_video_dates(individual_code, csv_folder):
        csv_files = [f for f in sorted(os.listdir(csv_folder)) if f.endswith('.csv')]
        ret = []
        for f in csv_files:
            date, species, zoo, individual, _, _ = f.split('_')
            if '{}_{}_{}'.format(species, zoo, individual) == individual_code:
                ret.append(date)
        return ret

    ret = []
    for j in range(len(spec)):
        ind_code = '{}_{}_{}'.format(spec[j], zoo[j], ind[j])
        csv_f = csv_folder[j]
        dates = _get_video_dates(ind_code, csv_f)
        ret.append(dates)
    return ret
        
        


        
def ensure_directory(filename):
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)


def get_accuracy_numbers(prediction_real_data, prediction_ai):
    
    minlen = min(len(prediction_real_data), len(prediction_ai))
    prediction_real_data = prediction_real_data[0:minlen]
    prediction_ai = prediction_ai[0:minlen]
    
    precision, recall, fscore, _ = precision_recall_fscore_support(prediction_real_data, 
                                                                   prediction_ai, 
                                                                   labels = [j for j in range(len(BEHAVIORS))],
                                                                   zero_division = 0
                                                                   )
    acc = accuracy_score(prediction_real_data, prediction_ai)

    return acc, precision, recall, fscore

def get_statistics(wb, behavs = BEHAVIORS):
    ws = wb['Statistik']
    anzahl, dauer = [], []
    for j in range(len(behavs)):
        if ws.cell(2+j, 2).value:
            anzahl.append(int(ws.cell(2, 2+j).value))
            dauer.append(np.round( float(ws.cell(3, 2+j).value ),2))
        else:
            anzahl.append(0)
            dauer.append(0.0)

    
    return anzahl, dauer
   


def get_number_phases_and_duration(sequence, behavs = BEHAVIORS):
    
    def _extract_single_phases(np_array, timeinterval = INTERVAL_LENGTH):
        """
        
    
        Parameters
        ----------
        np_array : TYPE
            DESCRIPTION.
        timeinterval : TYPE, optional
            DESCRIPTION. The default is cf.INTERVAL_LENGTH.
    
        Returns
        -------
        phases : [ [phase_len, phase_behavior, phase_start_interval, phase_end_interval] ]
    
        """
        phases = []
        LastBehav = np_array[0]
        iCurrLen = 1
        j = 1
        start_interval = 1

        for DistLine in np_array:
            
            if LastBehav == DistLine:
                iCurrLen += 1
            else:
                phases.append([iCurrLen*timeinterval, LastBehav, start_interval, j-1])
                iCurrLen = 1
                start_interval = j
                LastBehav = DistLine
            j += 1

        phases.append([iCurrLen*timeinterval, LastBehav, phases[-1][3] + 1, phases[-1][3] + 1 + iCurrLen])
        return phases
    
    
    num_phases = {}
    duration = {}
    median_len = {}
    phase_list = _extract_single_phases(sequence)
    total_dur = 0
    for j in range( len(behavs) ):
        num_phases[j] = len([ p for p in phase_list if p[1] == j ])
        duration[j] = np.sum( [ p[0] for p in phase_list if p[1] == j ] ) 
        total_dur += duration[j]
        med = np.median( [ p[0] for p in phase_list if p[1] == j ] )
        median_len[j] = 0 if np.isnan(med) else med
    for j in range( len(behavs) ):
        duration[j] /= total_dur
    return num_phases, duration, median_len
        




def compare_one_night(seq_orig, seq_prediction):
    
 
    anzahl_vid, dauer_vid, median_orig = get_number_phases_and_duration(seq_orig)
    anzahl_ai, dauer_ai, median_ai = get_number_phases_and_duration(seq_prediction)
    acc, _, _, f_score = get_accuracy_numbers(seq_orig, seq_prediction)
    
    print("Accuracy:", acc)
    print("F-Score:", f_score)

    return anzahl_vid, anzahl_ai, dauer_vid, dauer_ai, median_orig, median_ai, acc, f_score




def describe_errors(real_sequence, ai_sequence):
    
    i_min = min(len(real_sequence), len(ai_sequence))
    wrong_indices = [i for i in range(i_min) if real_sequence[i] != ai_sequence[i]]
    
    wrong_phases = []
    
    conf_mat = confusion_matrix(real_sequence[:i_min], ai_sequence[:i_min], labels = [i for i in range(len(BEHAVIORS))])
    
    break_error = False
    for j in range(1, i_min):
        if j not in wrong_indices:
            break_error = True
            continue
        if len(wrong_phases) == 0 or break_error:
            wrong_phases.append({'start': j, 'end': j, 
                                 'ai': ai_sequence[j], 
                                 'real': real_sequence[j]})
            break_error = False
        else:
            if real_sequence[j] == wrong_phases[-1]['real'] and ai_sequence[j] == wrong_phases[-1]['ai']:
                wrong_phases[-1]['end'] = j
            else:
                wrong_phases.append({'start': j, 'end': j, 
                                 'ai': ai_sequence[j], 
                                 'real': real_sequence[j]})
    
    return wrong_phases, conf_mat
                
   


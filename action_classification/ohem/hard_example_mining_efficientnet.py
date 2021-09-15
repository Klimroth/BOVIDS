#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

import os
import shutil
import csv, random
import numpy as np
import pandas as pd

KI_AUSWERTUNG = ''
KI_CUTOUT = ''

OUTPUT_BASE = ''
RANDOM_IMAGES = True
IMAGES_PER_INDIVIDUAL_AND_CLASS = 25

CUT_OFF_SECONDS = int(60*60*14) - 14
CUT_OFF_INTERVALS = CUT_OFF_SECONDS // 7 + 3

OFFSET_MF_CORRECTION = 1

INDIVIDUALS = []



if not RANDOM_IMAGES:
    print("Mining hard examples.")
    
    MAX_CONFIDENCY_RANDOM_IMAGES = 0.75
    MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL_AND_CLASS = int(IMAGES_PER_INDIVIDUAL_AND_CLASS*3/4)
    MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL = MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL_AND_CLASS*3
else:
    print("Creating random images.")
    MAX_CONFIDENCY_RANDOM_IMAGES = 1.01    
    MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL = 0
    MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL_AND_CLASS = 0














def _ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)



def add_ensemble_value(input_folder, output_folder):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    for csvf in csv_files:
        _add_ensemble_prediction(csvf, output_folder)


def _add_ensemble_prediction(stats_csv, outputfolder, auswertungsordner = KI_AUSWERTUNG):
    
    _ensure_dir(outputfolder)
    
    individuals = []
    with open(stats_csv, 'r') as csv_f:
        j = 1
        for row in csv_f.readlines():
            if j == 1:
                continue
            indcode = row[3]
            if not indcode in individuals:
                individuals.append(indcode)        
            j += 1

    overview_sheet_ens = {}
    for individual_code in individuals:
        
        species, zoo, num = individual_code.split('_')
        path_to_individualfolder_pred = auswertungsordner + species + '/' + zoo + '/' + num + '/'
        
        csv_folder_ens =  '{}total/raw_csv/ensemble/'.format(path_to_individualfolder_pred)
        if not os.path.exists(csv_folder_ens):
            print("Error (Path):", csv_folder_ens)
            return
        
    
        for csv_file in os.listdir(csv_folder_ens):
            date = csv_file.split('_')[0]
            _read_csv(path_to_csv = csv_folder_ens + csv_file, datum = date, 
                      individualcode = individual_code, 
                      return_dict = overview_sheet_ens)
        
    csvf = pd.read_csv(stats_csv)
    csvf['label_ens'] = ''
    
    for j in range(len(csvf)):
        if j == 0:
            continue
        mf_img = csvf.loc[j, 'relative_path_mf']
        interval = int(mf_img.split('/')[-1][:-4])
        datum = csvf.loc[j, 'identifier'].split('_')[0]
        individual = csvf.loc[j, 'individual']
        
        ens_label = _get_mf_label(individual_code = individual, 
                                                 date = datum, 
                                                 img_name = interval, 
                                                 mf_overview = overview_sheet_ens)
        csvf.loc[j,'label_ens'] = ens_label
    
    csvf.to_csv(outputfolder + stats_csv, index=False)
        
def _read_csv(path_to_csv, datum, individualcode, return_dict, mode,
              cut_sf = CUT_OFF_SECONDS, cut_mf = CUT_OFF_INTERVALS,
              max_confidency = MAX_CONFIDENCY_RANDOM_IMAGES):
    
    """ csv_format: image_name	startframe	endframe	standing	lying	sleeping	out
    Returns {individualcode: {  datum: { 0:[list of img-names], 1:[], 2:[]}  }}
    
    """
    
    if not os.path.exists(path_to_csv):
        print("Error: File does not exist.", path_to_csv)
        return

    if not individualcode in return_dict.keys():
        return_dict[individualcode] = {}
    if not datum in return_dict[individualcode].keys():
        return_dict[individualcode][datum] = {0:[], 1:[], 2:[]}
    
    cut = cut_sf if mode == 'sf' else cut_mf
    with open(path_to_csv, "r") as f:

        csv_real = csv.reader(f, delimiter=',')
        j = 0    
        for row in csv_real:
            j += 1
            if j == 1:
                continue
            
            if j > cut:
                break
            
            curr_behav = np.argmax([ np.float(row[3]), np.float(row[4]), np.float(row[5]), np.float(row[6]) ])
            conf = np.max([ np.float(row[3]), np.float(row[4]), np.float(row[5]), np.float(row[6]) ] )
            if curr_behav == 3 or conf >= max_confidency:
                continue
            
            if not str(row[0]).endswith('.jpg'):
                if mode == 'sf':
                    return_dict[individualcode][datum][curr_behav].append(str(row[0]).zfill(7) + '.jpg') 
                elif mode == 'mf':
                    return_dict[individualcode][datum][curr_behav].append(str(int(row[0])-OFFSET_MF_CORRECTION ).zfill(7) + '.jpg') 
            else:
                return_dict[individualcode][datum][curr_behav].append(row[0])


def _get_all_labels_individual(path_to_individualfolder_pred,
                               path_to_individualfolder_cutout,
                               overview_dictionary, overview_dictionary_mf,
                               individual_code, confidency):
    if not os.path.exists(path_to_individualfolder_pred):
        print("Error (Path):", path_to_individualfolder_pred)
        return
    
    if not os.path.exists(path_to_individualfolder_cutout):
        print("Error (Path):", path_to_individualfolder_cutout)
        return
    
    csv_folder =  '{}total/raw_csv/single_frames/'.format(path_to_individualfolder_pred)
    if not os.path.exists(csv_folder):
        print("Error (Path):", csv_folder)
        return
    
    csv_folder_mf =  '{}total/raw_csv/multiple_frames/'.format(path_to_individualfolder_pred)
    if not os.path.exists(csv_folder_mf):
        print("Error (Path):", csv_folder_mf)
        return
    

    for csv_file in os.listdir(csv_folder):
        date = csv_file.split('_')[0]
        _read_csv(path_to_csv = csv_folder + csv_file, datum = date, 
                  individualcode = individual_code, 
                  return_dict = overview_dictionary,
                  mode = 'sf',
                  max_confidency=confidency)
    
    for csv_file in os.listdir(csv_folder_mf):
        date = csv_file.split('_')[0]
        _read_csv(path_to_csv = csv_folder_mf + csv_file, datum = date, 
                  individualcode = individual_code, 
                  return_dict = overview_dictionary_mf,
                  mode = 'mf',
                  max_confidency=confidency)

def _sample_images_from_overview(overview, conflicts, sampled, low_confidency,
                                 num = IMAGES_PER_INDIVIDUAL_AND_CLASS,
                                 num_conflicts = MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL,
                                 num_conflicts_per_class = MAXIMUM_NUMBER_CONFLICTS_PER_INDIVIDUAL_AND_CLASS
                                 ):
    
    
    for individual in conflicts.keys():
        print('**************************************')
        print(individual)
        if not individual in sampled.keys():
            sampled[individual] = {}
            
        stehen, liegen, schlafen = 0, 0, 0
        stehen_a, liegen_a, schlafen_a = 0,0,0
        stehen_c, liegen_c, schlafen_c = 0,0,0
        
        for datum in conflicts[individual].keys():
            stehen += len(conflicts[individual][datum][0])
            liegen += len(conflicts[individual][datum][1])
            schlafen += len(conflicts[individual][datum][2])
            
            stehen_a += len(overview[individual][datum][0])
            liegen_a += len(overview[individual][datum][1])
            schlafen_a += len(overview[individual][datum][2])
            
            stehen_c += len(low_confidency[individual][datum][0])
            liegen_c += len(low_confidency[individual][datum][1])
            schlafen_c += len(low_confidency[individual][datum][2])
        
        conflict_count = [stehen, liegen, schlafen]
        behavior_count = [stehen_a, liegen_a, schlafen_a]
        low_confidency_count = [stehen_c, liegen_c, schlafen_c]
        
        print("Behavior class sizes:", behavior_count)
        print("Conflicts:", conflict_count)
        print("Low confidency:", low_confidency_count)
         
        perc = min(1.0, num_conflicts/(stehen + liegen + schlafen))
        img_conflicts_b = [int(stehen*perc), int(liegen*perc), int(schlafen*perc)]

        sample_count = [0,0,0]
        for datum in conflicts[individual].keys():
            if not datum in sampled[individual].keys():
                sampled[individual][datum] = { 0:[], 1:[], 2:[] }
            for b in [0,1,2]:
                if conflict_count[b] == 0:
                    continue
                per_date = min(int(img_conflicts_b[b] * len(conflicts[individual][datum][b]) / conflict_count[b]),
                               int(num_conflicts_per_class * len(conflicts[individual][datum][b]) / conflict_count[b]),
                               len(conflicts[individual][datum][b]))
                imgs = random.sample(conflicts[individual][datum][b], per_date)
                for img in imgs:
                    sampled[individual][datum][b].append(img)
                    sample_count[b] += 1
        
        print("Chosen conflicts:", sample_count)
        
        
        sample_count_c = [0,0,0]
        num_left = [max(0, num-sample_count[b]) for b in [0,1,2]]
        for datum in low_confidency[individual].keys():

            for b in [0,1,2]:
                if low_confidency_count[b] == 0:
                    continue
                per_date = min(int(num_left[b] * len(low_confidency[individual][datum][b]) / low_confidency_count[b]), len(low_confidency[individual][datum][b]))
                
                imgs = random.sample(low_confidency[individual][datum][b], per_date)
                for img in imgs:
                    sampled[individual][datum][b].append(img)
                    sample_count_c[b] += 1
        
        print("Additional samples (low confidency):", sample_count_c)
        
        
        num_left = [max(0, num-sample_count[b]-sample_count_c[b]) for b in [0,1,2]]
        sample_count_a = [0,0,0]
        for datum in overview[individual].keys():

            for b in [0,1,2]:
                if behavior_count[b] == 0:
                    continue
                per_date = min(int(num_left[b] * len(overview[individual][datum][b]) / behavior_count[b])+1, len(overview[individual][datum][b]))
                imgs = random.sample(overview[individual][datum][b], per_date)
                for img in imgs:
                    if sample_count_a[b] > num_left[b]:
                        continue
                    sampled[individual][datum][b].append(img)
                    sample_count_a[b] += 1
                    
                    
        
        print("Additional samples (to balance classes):", sample_count_a)
        
        
        

def _get_mf_label(individual_code, date, img_name, mf_overview):
    if not individual_code in mf_overview.keys():
        return 3
    if not date in mf_overview[individual_code].keys():
        return 3
    if img_name in mf_overview[individual_code][date][1]:
        return 1
    if img_name in mf_overview[individual_code][date][0]:
        return 0
    if img_name in mf_overview[individual_code][date][2]:
        return 2
    return 3

def _write_samples(sampled, out, mf_overview):
    
    _ensure_dir(out)
    
    for individual in sampled.keys():
        csv_path = out + individual + '.csv'  
        
        if os.path.exists(csv_path):
            os.remove(csv_path)
            
        with open(csv_path, 'w+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['relative_path_cutout', 'relative_path_mf', 'identifier_sf', 'identifier_mf', 'individual', 'label_ai', 'label_human', 'label_mf'])
            
            for datum in sampled[individual].keys():
                for behav in sorted(sampled[individual][datum].keys()):
                    for img in sampled[individual][datum][behav]:
                        rel_p = '{}/single_frames/{}/0/{}'.format(individual, datum, img)  
                        
                        frame_num = int(img.split('.')[0])
                        interval_num = frame_num // 7 + 1
                        interval = str(interval_num).zfill(7) + '.jpg'
                        rel_p_mf = '{}/multiple_frames/{}/0/{}'.format(individual, datum, interval)  
                        identifier_sf = '{}_{}_{}'.format(datum, individual, img)
                        identifier_mf = '{}_{}_{}'.format(datum, individual, interval)
                        
                        mf_label = _get_mf_label(individual_code = individual, 
                                                 date = datum, 
                                                 img_name = interval, 
                                                 mf_overview = mf_overview)
                        spamwriter.writerow([rel_p, rel_p_mf, identifier_sf, identifier_mf, individual, behav, 99, mf_label])
        
            
 
        
def _get_conflicts( conflict_sheet_sf, conflict_sheet_mf, overview_sheet, overview_sheet_mf ):
    
    for individual in overview_sheet.keys():
        if not individual in conflict_sheet_sf.keys():
            conflict_sheet_sf[individual] = {}
            conflict_sheet_mf[individual] = {}
            
        for datum in overview_sheet[individual].keys():
            
            if not datum in conflict_sheet_sf[individual].keys():
                conflict_sheet_sf[individual][datum] = {}
                conflict_sheet_mf[individual][datum] = {}
            
            for behav in overview_sheet[individual][datum].keys():
                
                if not behav in conflict_sheet_sf[individual][datum].keys():
                    conflict_sheet_sf[individual][datum][behav] = []
                    conflict_sheet_mf[individual][datum][behav] = []
                
                for img in overview_sheet[individual][datum][behav]:
                    frame_num = int(img.split('.')[0])
                    interval_num = frame_num // 7 + 1
                    interval = str(interval_num).zfill(7) + '.jpg'
                    
                    mf_label = _get_mf_label(individual_code = individual, 
                                                 date = datum, 
                                                 img_name = interval, 
                                                 mf_overview = overview_sheet_mf)
                    
                    if int(behav) == int(mf_label):
                        continue
                    
                    conflict_sheet_sf[individual][datum][behav].append(img)
                    conflict_sheet_mf[individual][datum][behav].append(interval)
                    
                    
    
    
    
def get_samples(auswertungsordner = KI_AUSWERTUNG, 
                        individuals = INDIVIDUALS,
                        cutoutordner = KI_CUTOUT,
                        out = OUTPUT_BASE,
                        confidency = MAX_CONFIDENCY_RANDOM_IMAGES):
    
    overview_sheet = {}
    overview_sheet_mf = {}
    
    low_confidency_sheet = {}
    low_confidency_sheet_mf = {}
    
    conflict_sheet_sf = {}
    conflict_sheet_mf = {}
    
    sample_sheet = {}
    
    for individual in individuals:
        print("Starting to sample:", individual)
        species, zoo, num = individual.split('_')
        if not species in os.listdir(auswertungsordner):
            print("No valid species:", individual)
            continue
        
        if not zoo in os.listdir(auswertungsordner + species):
            print("No valid zoo:", individual)
            continue
        
        if not num in os.listdir(auswertungsordner + species + '/' + zoo):
            print("No valid individual_number:", individual)
            continue
        
        if not individual in os.listdir(cutoutordner):
            print("No valid cut-out folder:", individual)
            continue

        _get_all_labels_individual(path_to_individualfolder_pred = auswertungsordner + species + '/' + zoo + '/' + num + '/',
                               path_to_individualfolder_cutout = cutoutordner + individual + '/',
                               overview_dictionary = overview_sheet,
                               overview_dictionary_mf=overview_sheet_mf,
                               individual_code = individual,
                               confidency = 2.0)
        
        _get_all_labels_individual(path_to_individualfolder_pred = auswertungsordner + species + '/' + zoo + '/' + num + '/',
                               path_to_individualfolder_cutout = cutoutordner + individual + '/',
                               overview_dictionary = low_confidency_sheet,
                               overview_dictionary_mf=low_confidency_sheet_mf,
                               individual_code = individual,
                               confidency = confidency)
    
    
    _get_conflicts(conflict_sheet_sf, conflict_sheet_mf, overview_sheet, overview_sheet_mf) 
    
    _sample_images_from_overview(overview_sheet, conflict_sheet_sf, sample_sheet, low_confidency_sheet)
    _write_samples(sample_sheet, out, overview_sheet_mf)

def move_images(sample_csv_folder = OUTPUT_BASE, 
                cutoutordner = KI_CUTOUT,
                individuals = INDIVIDUALS,
                out = OUTPUT_BASE):
    
    csv_files = [f for f in sorted(os.listdir(sample_csv_folder)) if f.endswith('.csv')]

    for csv_file in csv_files:
        individual = csv_file.split('.')[0]
        if not individual in individuals:
            continue
        print("Start copying images:", individual)
        
        if not individual in os.listdir(cutoutordner):
            print("No valid cut-out folder:", individual)
            continue
        
        dst_base_sf = out + 'SingleFrame/' + individual + '/'
        dst_base_mf = out + 'MultipleFrame/' + individual + '/'
        
        with open(sample_csv_folder + csv_file, "r") as f:
            csv_real = csv.reader(f, delimiter=',')
            j = 0    
            for row in csv_real:
                j += 1
                if j == 1:
                    continue
            
                img_rel_path_sf, img_rel_path_mf = row[0], row[1]
                img_name_sf, img_name_mf = row[2], row[3]
                
                b_sf = str(row[5])
                b_mf = str(row[7])
                
                src_sf = cutoutordner + img_rel_path_sf
                src_mf = cutoutordner + img_rel_path_mf
                
                if not os.path.exists(src_sf):
                    print("Error (File)", src_sf)
                    continue
                if not os.path.exists(src_mf):
                    print("Error (File)", src_mf)
                    continue
                
                dst_f_sf = dst_base_sf + b_sf + '/'
                dst_f_mf = dst_base_mf + b_mf + '/'
                _ensure_dir(dst_f_sf)
                _ensure_dir(dst_f_mf)
                
                dst_sf = dst_f_sf + img_name_sf
                dst_mf = dst_f_mf + img_name_mf
                shutil.copy(src_sf, dst_sf)
                shutil.copy(src_mf, dst_mf)
                
        
def get_samples_and_copy_images():
    get_samples()
    move_images()    
        
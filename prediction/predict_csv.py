#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.2"
__status__ = "Development"


AI_LIBRARY_LOCAL = 'prediction/'
AI_LIBRARY_GLOBAL = 'global/'
YOLO_LIBRARY = 'global/yolo-v4-tf.keras-master/'



import sys
sys.path.append(AI_LIBRARY_LOCAL)
sys.path.append(AI_LIBRARY_GLOBAL)
sys.path.append(YOLO_LIBRARY)

import os, csv, shutil, gc, pprint
import numpy as np
import tensorflow as tf

import configuration, image_creation_functions, behavior_prediction_functions, post_processing, image_cutout_functions
import prediction_density_object_detection as od_density

from datetime import datetime




"""
Initialise Tensorflow.
"""
# specify gpu to use ("0" or "1")
GPU_TO_USE = "0"


# Declare Error level and set precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_TO_USE
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

# TODO: test on z41 with multiple gpus
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('Used GPU:', gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.set_soft_device_placement(True)


tf.compat.v1.logging.set_verbosity(0)
tf.autograph.set_verbosity(0)



def predict_enclosure_by_csv(gpu_code = GPU_TO_USE,
                             skip_image_creation = configuration.SKIP_IMAGE_CREATION, 
                             
                             skip_individual_detection = configuration.SKIP_INDIVIDUAL_DETECTION,
                             
                             skip_single_frame_behavior_total = configuration.SKIP_BEHAVIOR_TOTAL_SF,                             
                             skip_multiple_frame_behavior_total = configuration.SKIP_BEHAVIOR_TOTAL_MF,
                             
                             skip_single_frame_behavior_binary = configuration.SKIP_BEHAVIOR_BINARY_SF,
                             skip_multiple_frame_behavior_binary = configuration.SKIP_BEHAVIOR_BINARY_MF,  
                             
                             skip_moving_files = configuration.SKIP_MOVING_FILES,  
                             skip_removing_temporary_files = configuration.SKIP_REMOVING_TEMPORARY_FILES,
                             
                             skip_post_processing_total = configuration.SKIP_PP_TOTAL,
                             skip_post_processing_binary = configuration.SKIP_PP_BINARY,
                             skip_od_density = configuration.SKIP_OD_DENSITY):
    
    
    _initialise_logging(input_csv = configuration.INPUT_CSV_FILE, 
                        output_folder=configuration.LOGGING_FILES)
    
    print('********************* Configuration ****************************')
    print('GPU:', gpu_code)
    print('Image creation:', 'conducted' if not skip_image_creation else 'skipped')
    print('Individual detection:', 'conducted' if not skip_individual_detection else 'skipped')
    print('Behavior (total) SF:', 'conducted' if not skip_single_frame_behavior_total else 'skipped')
    print('Behavior (total) MF:', 'conducted' if not skip_multiple_frame_behavior_total else 'skipped')
    print('Behavior (binary) SF:', 'conducted' if not skip_single_frame_behavior_binary else 'skipped')
    print('Behavior (binary) MF:', 'conducted' if not skip_multiple_frame_behavior_binary else 'skipped')
    print('Moving files:', 'conducted' if not skip_moving_files else 'skipped')
    print('Removing temporary files:', 'conducted' if not skip_removing_temporary_files else 'skipped')
    print('Post-Processing (total)', 'conducted' if not skip_post_processing_total else 'skipped')
    print('Post-Processing (binary)', 'conducted' if not skip_post_processing_binary else 'skipped')
    print('Calculate OD-Density', 'conducted' if not skip_od_density else 'skipped')
    print('****************************************************************')
    
    # load basic informations
    enclosure_info, individual_info = _get_enclosure_and_individuals()   
    if not enclosure_info:
        print("ERROR: Basic information could not be read from the CSV-file.")
        return
    
    #pp = pprint.PrettyPrinter(indent=4)
    
       
    
    if not skip_image_creation:
        try:                    
            tf.compat.v1.reset_default_graph()
            _ = gc.collect()
        except:
            print("WARNING: Garbage collection did not succeed.")  
        # create raw images
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 1: Create raw images from video files.")
        print("*****************************************************************")
        image_creation_functions.generate_raw_images()
   
    

    if not skip_individual_detection:
        try:                    
            tf.compat.v1.reset_default_graph()
            _ = gc.collect()
        except:
            print("WARNING: Garbage collection did not succeed.")  
        # Cut out images for each individual
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 2: Object Detection.")
        print("*****************************************************************")

        
        image_cutout_functions.predict_multiple_nights(enclosure_info, individual_info)
    
    
    
    if not skip_single_frame_behavior_total:
        try:                    
            tf.compat.v1.reset_default_graph()
            _ = gc.collect()
        except:
            print("WARNING: Garbage collection did not succeed.")  
            
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 3: Action Classification - Single Frame (Total).")
        print("*****************************************************************")
        
        for individual_code in sorted(individual_info.keys()):
            
            index = -1
            for datum in individual_info[individual_code]['dates']:
                index += 1
                input_folder_prediction = configuration.TMP_STORAGE_CUTOUT + individual_code + '/single_frames/' + datum + '/'
        
                if not os.path.exists(input_folder_prediction):
                    print("ERROR: Folder does not exist.", input_folder_prediction)
                    continue
            
                try:                    
                    tf.compat.v1.reset_default_graph()
                    _ = gc.collect()
                except:
                    print("WARNING: Garbage collection did not succeed.")        
                
                number_frames_video = 3600*_get_video_length(individual_info[individual_code]['start_times'][index], individual_info[individual_code]['end_times'][index])
                species, zoo, individual = individual_code.split('_')
                
                
                output_folder_prediction = configuration.FINAL_STORAGE_PREDICTION_FILES + species + '/' + zoo + '/' + individual + '/'
                behavior_prediction_functions.predict_folder_single_frames(folder_path = input_folder_prediction,
                                                                           individual_code = individual_code, 
                                                                           datum = datum,
                                                                           amount_frames=number_frames_video,
                                                                           output_folder = output_folder_prediction,
                                                                           type_of_behavior = 'total')
                

    
    if not skip_multiple_frame_behavior_total:
        try:                    
            tf.compat.v1.reset_default_graph()
            _ = gc.collect()
        except:
            print("WARNING: Garbage collection did not succeed.")  
            
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 4: Action Classification - Multiple Frame (Total).")
        print("*****************************************************************")
        # same game with joint images
        # iterating through folder structure, each prediction (each night) gets a fresh session
        for individual_code in sorted(individual_info.keys()):
            
            index = -1
            for datum in individual_info[individual_code]['dates']:
                index += 1
                input_folder_prediction = configuration.TMP_STORAGE_CUTOUT + individual_code + '/multiple_frames/' + datum + '/'
        
                if not os.path.exists(input_folder_prediction):
                    print("ERROR: Folder does not exist.", input_folder_prediction)
                    continue
            
                try:                    
                    tf.compat.v1.reset_default_graph()
                    _ = gc.collect()
                except:
                    print("WARNING: Garbage collection did not succeed.")        
                
                number_frames_video = 3600*_get_video_length(individual_info[individual_code]['start_times'][index], individual_info[individual_code]['end_times'][index])
                
                species, zoo, individual = individual_code.split('_')
                
                
                output_folder_prediction = configuration.FINAL_STORAGE_PREDICTION_FILES + species + '/' + zoo + '/' + individual + '/'
                behavior_prediction_functions.predict_folder_multiple_frames(folder_path = input_folder_prediction,
                                                                           individual_code = individual_code, 
                                                                           datum = datum,
                                                                           amount_frames=number_frames_video,
                                                                           output_folder = output_folder_prediction,
                                                                           type_of_behavior = 'total')
                
    if not skip_single_frame_behavior_binary:
        try:                    
            tf.compat.v1.reset_default_graph()
            _ = gc.collect()
        except:
            print("WARNING: Garbage collection did not succeed.")  
            
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 5: Action Classification - Single Frame (Binary).")
        print("*****************************************************************")
        
        for individual_code in sorted(individual_info.keys()):
            
            index = -1
            for datum in individual_info[individual_code]['dates']:
                index += 1
                input_folder_prediction = configuration.TMP_STORAGE_CUTOUT + individual_code + '/single_frames/' + datum + '/'
        
                if not os.path.exists(input_folder_prediction):
                    print("ERROR: Folder does not exist.", input_folder_prediction)
                    continue
            
                try:                    
                    tf.compat.v1.reset_default_graph()
                    _ = gc.collect()
                except:
                    print("WARNING: Garbage collection did not succeed.")        
                
                number_frames_video = 3600*_get_video_length(individual_info[individual_code]['start_times'][index], individual_info[individual_code]['end_times'][index])
                species, zoo, individual = individual_code.split('_')
                
                
                output_folder_prediction = configuration.FINAL_STORAGE_PREDICTION_FILES + species + '/' + zoo + '/' + individual + '/'
                behavior_prediction_functions.predict_folder_single_frames(folder_path = input_folder_prediction,
                                                                           individual_code = individual_code, 
                                                                           datum = datum,
                                                                           amount_frames=number_frames_video,
                                                                           output_folder = output_folder_prediction,
                                                                           type_of_behavior = 'binary')
                

    
    if not skip_multiple_frame_behavior_binary:
        try:                    
            tf.compat.v1.reset_default_graph()
            _ = gc.collect()
        except:
            print("WARNING: Garbage collection did not succeed.")  
            
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 6: Action Classification - Multiple Frame (Binary).")
        print("*****************************************************************")
        # same game with joint images
        # iterating through folder structure, each prediction (each night) gets a fresh session
        for individual_code in sorted(individual_info.keys()):
            
            index = -1
            for datum in individual_info[individual_code]['dates']:
                index += 1
                input_folder_prediction = configuration.TMP_STORAGE_CUTOUT + individual_code + '/multiple_frames/' + datum + '/'
        
                if not os.path.exists(input_folder_prediction):
                    print("ERROR: Folder does not exist.", input_folder_prediction)
                    continue
            
                try:                    
                    tf.compat.v1.reset_default_graph()
                    _ = gc.collect()
                except:
                    print("WARNING: Garbage collection did not succeed.")        
                
                number_frames_video = 3600*_get_video_length(individual_info[individual_code]['start_times'][index], individual_info[individual_code]['end_times'][index])
                
                species, zoo, individual = individual_code.split('_')
                
                
                output_folder_prediction = configuration.FINAL_STORAGE_PREDICTION_FILES + species + '/' + zoo + '/' + individual + '/'
                behavior_prediction_functions.predict_folder_multiple_frames(folder_path = input_folder_prediction,
                                                                           individual_code = individual_code, 
                                                                           datum = datum,
                                                                           amount_frames=number_frames_video,
                                                                           output_folder = output_folder_prediction,
                                                                           type_of_behavior = 'binary')            
    
    try:                    
        tf.compat.v1.reset_default_graph()
        _ = gc.collect()
    except:
        print("WARNING: Garbage collection did not succeed.")  
        
    if not skip_moving_files:
        
        if configuration.TMP_STORAGE_CUTOUT == configuration.FINAL_STORAGE_CUTOUT:
            print("Temporary storage and final storage coincide - skipped moving files.")
        else:
        
            print("************************************************************************************")
            print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 7: Moving files to final storage")
            print("************************************************************************************")
            root_src_dir = configuration.TMP_STORAGE_CUTOUT
            root_dst_dir = configuration.FINAL_STORAGE_CUTOUT
            
            
            for individual_code in sorted(individual_info.keys()):
                mf_src_path = root_src_dir + individual_code + '/multiple_frames/'
                sf_src_path = root_src_dir + individual_code + '/single_frames/'
                pf_src_path = root_src_dir + individual_code + '/position_files/'
                
                for datum in individual_info[individual_code]['dates']:
                    date_mf = mf_src_path + datum + '/'
                    date_sf = sf_src_path + datum + '/'
                    date_pf = pf_src_path + datum + '/'
                    
                    if not os.path.exists(date_sf):
                        print("ERROR: Path not found:", date_sf)
                        continue
                    if not os.path.exists(date_mf):
                        print("ERROR: Path not found:", date_mf)
                        continue
                    if not os.path.exists(date_sf):
                        print("ERROR: Path not found:", date_mf)
                        continue
                    
                    
                    print(datetime.now().strftime('%Y-%m-%d %H:%M'), individual_code, datum)
                    mf_target = root_dst_dir + individual_code + '/multiple_frames/' + datum + '/'
                    sf_target = root_dst_dir + individual_code + '/single_frames/' + datum + '/'
                    pf_target = root_dst_dir + individual_code + '/position_files/' + datum + '/'
                    
                    ensure_and_delete_directory(mf_target)
                    ensure_and_delete_directory(sf_target)
                    ensure_and_delete_directory(pf_target)
                    
                    shutil.copytree(src=date_mf, dst=mf_target, dirs_exist_ok=True)
                    shutil.copytree(src=date_sf, dst=sf_target, dirs_exist_ok=True)
                    shutil.copytree(src=date_pf, dst=pf_target, dirs_exist_ok=True)                    
            
                    if not skip_removing_temporary_files:
                        shutil.rmtree(date_mf, ignore_errors = True)
                        shutil.rmtree(date_sf, ignore_errors = True)
                        shutil.rmtree(date_pf, ignore_errors = True)
            
            if not skip_removing_temporary_files:
                for enclosure_code in sorted(enclosure_info.keys()):
                    for datum in enclosure_info[enclosure_code]['dates']:
                        curr_path = '{}{}/{}'.format(configuration.TMP_STORAGE_IMAGES, enclosure_code, datum)
                        if not os.path.exists(curr_path):
                            continue                    
                        shutil.rmtree(curr_path, ignore_errors = True)
       
        
    
        
    if not skip_post_processing_total:        
        
        
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 8.1: Post-Processing (total).")
        print("*****************************************************************")
        
        
        for individual_code in sorted(individual_info.keys()):
            species, zoo, individual_num = individual_code.split('_')
            
            base_path = '{}{}/{}/{}/'.format(configuration.FINAL_STORAGE_PREDICTION_FILES, species, zoo, individual_num)
            if not os.path.exists(base_path):
                print("ERROR: Path not found:", base_path)
                continue
            
            index = -1
            logfile_truncation = base_path + individual_code + '_logfile_truncation.txt'
            if os.path.exists(logfile_truncation):
                os.remove(logfile_truncation)

            for datum in individual_info[individual_code]['dates']:
                index += 1
                
                pos_file_folder = '{}{}/position_files/{}/'.format(configuration.FINAL_STORAGE_CUTOUT, individual_code, datum)
                if not os.path.exists(pos_file_folder):
                    print("ERROR: Path not found:", pos_file_folder)
                    continue
                
                mf_csv = '{}/total/raw_csv/multiple_frames/{}_{}.csv'.format(base_path, datum, individual_code)
                sf_csv = '{}/total/raw_csv/single_frames/{}_{}.csv'.format(base_path, datum, individual_code)
                
                
                if not os.path.exists(mf_csv):
                    print("ERROR: File not found:", mf_csv)
                    continue
                if not os.path.exists(sf_csv):
                    print("ERROR: File not found:", sf_csv)
                    continue
                
                post_processing_key = individual_info[individual_code]['postprocessors'][index]
                
                if not post_processing_key in configuration.POST_PROCESSING_RULES.keys():
                    print('ERROR: {} is not a valid post-processing rule.'.format(post_processing_key))
                    continue
                print(datetime.now().strftime('%Y-%m-%d %H:%M'), individual_code, datum)
                
                post_processing.post_process_night(single_frame_csv = sf_csv, 
                                               joint_interval_csv = mf_csv, 
                                               individual_code = individual_code, 
                                               output_folder_prediction = base_path + 'total/',
                                               behavior_mapping = {0:0, 1:1, 2:2, 3:3, 4:4},
                                               datum = datum, 
                                               position_files = pos_file_folder, 
                                               is_test = False,
                                               post_processing_rules = configuration.POST_PROCESSING_RULES[post_processing_key],
                                               truncation_rules = individual_info[individual_code]['truncations'][index],
                                               current_video_start=individual_info[individual_code]['start_times'][index],
                                               current_video_end=individual_info[individual_code]['end_times'][index],
                                               logfile = logfile_truncation )
                

    
    if not skip_post_processing_binary:        
        
        
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 8.2: Post-Processing (binary).")
        print("*****************************************************************")
        
        
        for individual_code in sorted(individual_info.keys()):
            species, zoo, individual_num = individual_code.split('_')
            
            base_path = '{}{}/{}/{}/'.format(configuration.FINAL_STORAGE_PREDICTION_FILES, species, zoo, individual_num)
            if not os.path.exists(base_path):
                print("ERROR: Path not found:", base_path)
                continue
            
            index = -1
            logfile_truncation_binary = base_path + individual_code + '_logfile_truncation_binary.txt'
            if os.path.exists(logfile_truncation_binary):
                os.remove(logfile_truncation_binary)

            for datum in individual_info[individual_code]['dates']:
                index += 1
                
                pos_file_folder = '{}{}/position_files/{}/'.format(configuration.FINAL_STORAGE_CUTOUT, individual_code, datum)
                if not os.path.exists(pos_file_folder):
                    print("ERROR: Path not found:", pos_file_folder)
                    continue
                
                mf_csv = '{}/binary/raw_csv/multiple_frames/{}_{}.csv'.format(base_path, datum, individual_code)
                sf_csv = '{}/binary/raw_csv/single_frames/{}_{}.csv'.format(base_path, datum, individual_code)
                
                
                if not os.path.exists(mf_csv):
                    print("ERROR: File not found:", mf_csv)
                    continue
                if not os.path.exists(sf_csv):
                    print("ERROR: File not found:", sf_csv)
                    continue
                
                post_processing_key_b = individual_info[individual_code]['postprocessors-binary'][index]
                if not post_processing_key_b in configuration.POST_PROCESSING_RULES.keys():
                    print('ERROR: {} is not a valid post-processing rule.'.format(post_processing_key_b))
                    continue
                print(datetime.now().strftime('%Y-%m-%d %H:%M'), individual_code, datum)

                post_processing.post_process_night(single_frame_csv = sf_csv, 
                                                joint_interval_csv = mf_csv, 
                                                individual_code = individual_code, 
                                                output_folder_prediction = base_path + 'binary/',
                                                behavior_mapping = {0:0, 1:1, 2:1, 3:3, 4:4},
                                                datum = datum, 
                                                position_files = pos_file_folder, 
                                                is_test = False,
                                                extension = '_binary',
                                                post_processing_rules = configuration.POST_PROCESSING_RULES[post_processing_key_b],
                                                truncation_rules = individual_info[individual_code]['truncations'][index],
                                                current_video_start=individual_info[individual_code]['start_times'][index],
                                                current_video_end=individual_info[individual_code]['end_times'][index],
                                                logfile = logfile_truncation_binary)
                    
                    
                
        
        
    if not skip_od_density:
        print("*****************************************************************")
        print(datetime.now().strftime('%Y-%m-%d %H:%M'), "- Step 9: Detection Density.")
        print("*****************************************************************")
        
        
        for individual_code in sorted(individual_info.keys()):
            sequences = {}
            sequences[individual_code] =  {}
            species, zoo, individual_num = individual_code.split('_')
            
            base_path = '{}{}/{}/{}/'.format(configuration.FINAL_STORAGE_PREDICTION_FILES, species, zoo, individual_num)
            if not os.path.exists(base_path):
                print("ERROR: Path not found:", base_path)
                continue
            
            index = -1
            logfile_truncation = base_path + individual_code + '_logfile_truncation.txt'
            if os.path.exists(logfile_truncation):
                os.remove(logfile_truncation)
            
            pos_file_folder_base = '{}/{}/position_files/'.format(configuration.FINAL_STORAGE_CUTOUT, individual_code)
            if not os.path.exists(pos_file_folder_base):
                print("ERROR: Path not found:", pos_file_folder_base)
                continue
        
            package_size = int(configuration.PACKAGE_SIZE_DENSITY_STATISTICS * (int(3600 / configuration.INTERVAL_LENGTH)+1))
            od_density_output = base_path + individual_code + '_Detection_Density_Overview.xlsx'
            
            index = -1
            
            for datum in individual_info[individual_code]['dates']:
                index += 1            
                
                if not os.path.isdir(pos_file_folder_base + datum):
                    print("WARNING: Folder does not exist.", pos_file_folder_base + datum)
                    continue
                
                max_intervals = (3600*_get_video_length(individual_info[individual_code]['start_times'][index], individual_info[individual_code]['end_times'][index]))
                max_intervals = int(np.ceil(max_intervals / configuration.INTERVAL_LENGTH))
                max_per_interval = configuration.IMAGES_PER_INTERVAL          
                                   
                total_nums, density, subsequence_nums, subsequence_density = od_density.get_sequence_per_night(pos_file_folder_base + datum + '/', max_intervals, package_size, max_per_interval)
                sequences[individual_code][datum] = [total_nums, density, subsequence_nums, subsequence_density]

            
            od_density.write_xlsx_overview(sequences[individual_code], od_density_output, individual_code)


    print("*****************************************************************")
    print("Process finished - ", datetime.now().strftime('%Y-%m-%d %H:%M'))
    print("*****************************************************************")
    
class Logger(object):
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.logfile_path = logfile_path

    def write(self, message):
        with open (self.logfile_path, "a", encoding = 'utf-8') as self.log:            
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        pass
  

def _initialise_logging(input_csv, output_folder):
    csv_name = input_csv.split('/')[-1].split('.')[0]
    log_file = output_folder + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + csv_name + '.txt'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)        
        
    f = open(log_file,"w+")
    f.close()
    
    
    sys.stdout = Logger(logfile_path=log_file)    

def _get_video_length(start, end):
    if end > start:
        return end-start
    
    return 24 + (end-start)

def ensure_and_delete_directory(dirpath):
    shutil.rmtree(dirpath, ignore_errors=True)
    os.makedirs(dirpath)

def _get_contained_structure(individual_info):
    ret = {}
    for individual_code in individual_info.keys():
        species, zoo, individual_num = individual_code.split('_')
        if not species in ret.keys():
            ret[species] = {}
        if not zoo in ret[species].keys():
            ret[species][zoo] = {}
        if not individual_num in ret[species][zoo].keys():
            ret[species][zoo][individual_num] = {'dates': [], 'start_times': [], 
                                                 'end_times':[], 'truncations':[], 
                                                 'postprocessors':[], 'postprocessors-binary':[], 'enclosures':[]}
        for k, v in individual_info[individual_code].items():
            ret[species][zoo][individual_num][k] = v
    
    return ret
        

def _get_enclosure_and_individuals(csv_filepath = configuration.INPUT_CSV_FILE):
    
    def _correct_date(date):
        if not "." in date:
            return date
        return date.split(".")[2] + "-" + date.split(".")[1] + "-" + date.split(".")[0]

    animal_sep = configuration.ANIMAL_NUMBER_SEPERATOR
    delim = configuration.CSV_DELIMITER
    
    if not os.path.exists(csv_filepath):
        print("Error: Input-CSV-file was not found.")
        return False, False
    
    enclosure_info = {}
    individual_info = {}
    

    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        line_count = 0
        for row in csv_reader:
            
            if line_count == 0:
                if len(row) != 14:
                    print("Overview file has the wrong format.")
                    return False, False
                else:
                    line_count += 1
                    continue
            enclosure_code = '{}_{}_{}'.format(row[1], row[2], row[3])
            individual_codes = ['{}_{}_{}'.format(row[1], row[2], x) for x in str(row[5]).split(animal_sep)]
            start_time, end_time = int(row[6]), int(row[7])
            if not enclosure_code in enclosure_info.keys():
                enclosure_info[enclosure_code] = {'dates': [], 'individuals': [], 'start_times': [], 'end_times': [], 
                                                  'truncations': [], 'postprocessors': [], 'postprocessors-binary':[]}
            
            
            enclosure_info[enclosure_code]['dates'].append(_correct_date(str(row[0])))
            enclosure_info[enclosure_code]['individuals'].append( individual_codes )
            enclosure_info[enclosure_code]['start_times'].append( start_time )
            enclosure_info[enclosure_code]['end_times'].append( end_time )
            enclosure_info[enclosure_code]['truncations'].append( {'up': int(row[10]), 'bot': int(row[11]), 'left': int(row[12]), 'right': int(row[13])} )
            enclosure_info[enclosure_code]['postprocessors'].append( row[8] )
            enclosure_info[enclosure_code]['postprocessors-binary'].append( row[9] )
            
            for individual_code in individual_codes:
                if not individual_code in individual_info.keys():
                    individual_info[individual_code] = {'dates': [], 'enclosures': [], 'start_times': [], 'end_times': [], 
                                                        'truncations': [], 'postprocessors': [], 'postprocessors-binary':[]}
                    
                individual_info[individual_code]['dates'].append(_correct_date(str(row[0])))
                individual_info[individual_code]['enclosures'].append( enclosure_code )
                individual_info[individual_code]['start_times'].append( start_time )
                individual_info[individual_code]['end_times'].append( end_time )
                individual_info[individual_code]['truncations'].append( {'up': int(row[10]), 'bot': int(row[11]), 'left': int(row[12]), 'right': int(row[13])} )
                individual_info[individual_code]['postprocessors'].append( row[8] )
                individual_info[individual_code]['postprocessors-binary'].append( row[9] )
            
            line_count += 1
        
        return enclosure_info, individual_info##


if __name__ == "__main__": 
    predict_enclosure_by_csv()
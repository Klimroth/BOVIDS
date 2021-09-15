#!/usr/bin/env python3
# -*- coding: iso-8859-1 -*-

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2021, M. Hahn-Klimroth, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes", "T. Kapetanopoulos"]
__license__ = "GPL-3.0"
__version__ = "1.0"
__status__ = "Development"

"""
Small functions that are used to work with the object detection training set.

extract_od_images_species()
    - ANNOTATION_FOLDER_IMAGES_ES: contains the images, starting with species/zoo/...
    e.g.: .../Annotation/Bilder/
    - ANNOTATION_FOLDER_LABELS_ES: contains the corresponding labels in the same
    structure, e.g. .../Annotation/Label/
    - OUTPUT_FOLDER_ES: output folder in which folders Bilder/ and Label/ will be created
    - SPECIES_ES: List of species: only the images and labels of the
    listed species's will be copied.
    - EXCLUDE_ENCLOSURES_ES, EXCLUDE_ZOOS_ES: Lists of enclosure codes and zoos which will be skipped.
    
    
extract_od_images_enclosures()
    - ANNOTATION_FOLDER_IMAGES_EE: contains the images, starting with species/zoo/...
    e.g.: .../Annotation/Bilder/
    - ANNOTATION_FOLDER_LABELS_EE: contains the corresponding labels in the same
    structure, e.g. .../Annotation/Label/
    - OUTPUT_FOLDER_EE: output folder in which folders Bilder/ and Label/ will be created
    - ENCLOSURES_EE: List of enclosure_codes: only the images and labels of the
    listed enclosures will be copied.
    

    
rename_annotation()
    - input: LABEL_RENAME_FOLDER: contains the .xml files in which the
    labels should be renamed.
    - input: OLD_LABEL: If not empty, it will only rename labels of OLD_LABEL
    - input: NEW_LABEL: Label to which the labels will be renamed.
    
extract_images_evaluated()

"""

import shutil, os
import xml.etree.ElementTree as ET


############# EXTRACT IMAGES FOR SPECIES ######################
ANNOTATION_FOLDER_IMAGES_ES = ''
ANNOTATION_FOLDER_LABELS_ES = ''
OUTPUT_FOLDER_ES = ''
SPECIES_ES = ['']
EXCLUDE_ENCLOSURES_ES = ['']
EXCLUDE_ZOOS_ES = ['']


############# EXTRACT IMAGES FROM EVALUATED IMAGES ######################
INPUT_FOLDER_BASE = ''
ENCLOSURE_CODE_EI = ''
WHICH_ONES = ['bad', 'gut', 'swapped']
OUTPUT_FOLDER_IMAGES_EI = ''
OUTPUT_FOLDER_LABELS_EI = ''

############# EXTRACT IMAGES FOR ENCLOSURES ######################
ANNOTATION_FOLDER_IMAGES_EE = ''
ANNOTATION_FOLDER_LABELS_EE = ''
OUTPUT_FOLDER_EE = ''
ENCLOSURES_EE = ['']


############# RENAME XML CLASSES ######################
LABEL_RENAME_FOLDER = ''
OLD_LABEL = '' # leave empty string if EVERY label should be renamed
NEW_LABEL = ''






def extract_images_evaluated(inp = INPUT_FOLDER_BASE,
                             enc = ENCLOSURE_CODE_EI,
                             which = WHICH_ONES,
                             outim = OUTPUT_FOLDER_IMAGES_EI,
                             outlab = OUTPUT_FOLDER_LABELS_EI):
    enc_struc = '{}/{}/{}/'.format( enc.split('_')[0], enc.split('_')[1], enc.split('_')[2] )
    ensure_directory(outim)
    ensure_directory(outlab)
    for subf in which:
        if not os.path.exists( inp + subf + '/Bilder/' + enc_struc ):
            print('Path not found:', inp + subf + '/Bilder/' + enc_struc)
            continue
        if not os.path.exists( inp + subf + '/Label/' + enc_struc ):
            print('Path not found:', inp + subf + '/Label/' + enc_struc)
            continue
        for img in os.listdir( inp + subf + '/Bilder/' + enc_struc ):            
            shutil.copy2( inp + subf + '/Bilder/' + enc_struc + img, outim + img )
        for lab in os.listdir( inp + subf + '/Label/' + enc_struc ):            
            shutil.copy2( inp + subf + '/Label/' + enc_struc + lab, outlab + lab )
        


def ensure_directory(p):
    if not os.path.exists(p):
        os.makedirs(p)
        

def extract_od_images_species(img_base = ANNOTATION_FOLDER_IMAGES_ES, 
                              label_base = ANNOTATION_FOLDER_LABELS_ES,
                              output_folder = OUTPUT_FOLDER_ES,
                              species = SPECIES_ES,
                              ex_zoo = EXCLUDE_ZOOS_ES,
                              ex_enc = EXCLUDE_ENCLOSURES_ES):
    
    
    output_folder_img = output_folder + 'Bilder/'
    output_folder_label = output_folder + 'Label/'
    ensure_directory(output_folder_img)
    ensure_directory(output_folder_label)
    for spec in sorted(os.listdir( label_base )):
        if not spec in species:
            continue
        
        for zoo in sorted(os.listdir( label_base + spec )):
            if zoo in ex_zoo:
                continue
            
            for enc_num in sorted(os.listdir( label_base + spec + '/' + zoo)):
                enc_code = '{}_{}_{}'.format(spec, zoo, enc_num)
                if enc_code in ex_enc:
                    continue
                print('Currently processing: ', enc_code)
                for xml_file in sorted(os.listdir( label_base + spec + '/' + zoo + '/' + enc_num)):
                    if not xml_file.endswith('.xml'):
                        continue
                    
                    img_name = xml_file[:-4] + '.jpg'
                    src_label = label_base + spec + '/' + zoo + '/' + enc_num + '/' + xml_file
                    src_img = img_base + spec + '/' + zoo + '/' + enc_num + '/' + img_name
                    
                    dst_label = output_folder_label + xml_file
                    dst_img = output_folder_img + img_name
                    if not os.path.exists(src_img):
                        print('Warning: Image does not exist (skipped):', src_img)
                        continue
                    
                    shutil.copy2(src_label, dst_label)
                    shutil.copy2(src_img, dst_img)
                    
def extract_od_images_enclosures(img_base = ANNOTATION_FOLDER_IMAGES_EE, 
                              label_base = ANNOTATION_FOLDER_LABELS_EE,
                              output_folder = OUTPUT_FOLDER_EE,
                              enclosures = ENCLOSURES_EE):
    
    output_folder_img = output_folder + 'Bilder/'
    output_folder_label = output_folder + 'Label/'
    ensure_directory(output_folder_img)
    ensure_directory(output_folder_label)
    
    for spec in sorted(os.listdir( label_base )):
        
        for zoo in sorted(os.listdir( label_base + spec )):
            
            for enc_num in sorted(os.listdir( label_base + spec + '/' + zoo)):
                enc_code = '{}_{}_{}'.format(spec, zoo, enc_num)                
                if not enc_code in enclosures:
                    continue
                print('Currently processing: ', enc_code)
                for xml_file in sorted(os.listdir( label_base + spec + '/' + zoo + '/' + enc_num)):
                    if not xml_file.endswith('.xml'):
                        continue
                    
                    img_name = xml_file[:-4] + '.jpg'
                    src_label = label_base + spec + '/' + zoo + '/' + enc_num + '/' + xml_file
                    src_img = img_base + spec + '/' + zoo + '/' + enc_num + '/' + img_name
                    
                    dst_label = output_folder_label + xml_file
                    dst_img = output_folder_img + img_name
                    if not os.path.exists(src_img):
                        print('Warning: Image does not exist (skipped):', src_img)
                        continue
                    
                    shutil.copy2(src_label, dst_label)
                    shutil.copy2(src_img, dst_img)                    
                    


    
def rename_annotation(label_folder = LABEL_RENAME_FOLDER, 
                      old_label = OLD_LABEL,
                      new_label = NEW_LABEL):
    

    
    xml_list = [label_folder + x for x in sorted(os.listdir(label_folder)) if x.endswith(".xml")]
    for xml_file in xml_list:
        try:
            tree = ET.parse(xml_file, ET.XMLParser(encoding='latin1'))
            root = tree.getroot()
                
            for obj in root.findall('object'):
                name_tag = obj.find('name')
                if len(old_label) >= 1: 
                    if not name_tag.text == old_label:
                        continue
                name_tag.text = new_label
            tree.write(xml_file, encoding='latin1')
        except:
            print('Error: XML-file damaged - ', xml_file)
            #print('*File will be removed.')
            #os.remove(xml_file)



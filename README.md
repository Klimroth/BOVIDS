# BOVIDS
BOVIDS is an end-to-end deep learning based tool for posture estimation of ungulates in videos. It is the software presented and discussed in

>  
>  
>  

## License and citation
Re-use, distribution and modification as well as extending contributions under the MIT license are highly welcome. If you use BOVIDS or parts of it, please consider citing the following publications:

BOVIDS software package and an application:
>  
>  
>  

Technical analysis of the deep learning prediction pipeline:
>  Hahn‐Klimroth, M, Kapetanopoulos, T, Gübert, J, Dierkes, PW. Deep learning‐based pose estimation for African ungulates in zoos. 
>  
>  Ecol Evol. 2021; 00: 1– 18. https://doi.org/10.1002/ece3.7367 

## Installation
We suggest installing the necessary packages using anaconda as it will control the versions for you. 
>conda activate
>
>conda update conda anaconda
>
>conda create -n "bovids" python==3.8
>
>conda activate bovids
>
>conda install tensorflow-gpu==2.2 spyder keras openpyxl matplotlib tqdm pandas
>
>conda install -c conda-forge opencv efficientnet scikit-learn imgaug

If you are on linux, make sure to have the following packages installed.
> libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

Some parts of BOVIDS (i.e. if you want to use the tools for editing video files) require moviepy. As it is fairly likely to create a version clash between opencv, ffmpeg and moviepy, we suggest to create a different environment in that case.

>conda create -n "video" python==3.7
>
>conda activate video
>
>conda install spyder openpyxl matplotlib tqdm pandas
>
>conda install -c conda-forge moviepy opencv ffmpeg

Finally, the third party software (MIT license) [labelImg by tzutalin](https://github.com/tzutalin/labelImg) might be installed in a third environment. Detailed instructions can be found at the corresponding github repository.

## Data preparation and organisation

### Identifiers and organisation
BOVIDS requires a very specific data organisation to work flawless and comfortably. To this end, we suppose that we observe individuals of a specific **species** in multiple **zoos** in different **enclosures**. Each enclosure might be filmed by 1...n **video-streams** and, given a specific **date**, there are 1...m **individuals** in the enclosure. We suppose a date to have format YYYY-MM-DD and all names are not allowed to contain underscores. Then we define the following identifiers:

>*enclosurecode*: SPECIESNAME_ZOONAME_ENCLOSURENUMBER
>
>*individualcode*: SPECIESNAME_ZOONAME_INDIVIDUALNUMBER

If those identifiers are not unique over the all observation time, for instance, if video streams break or if individuals get stalled differently from time to time, the following identifiers are unique per night. A **night** is a collection of videos recording the same enclosure at a specific date with a starting time (standard 5 p.m.) and an ending time (standard 7 a.m.), these values are variable.

>*enclosureindividualcode*: SPECIESNAME_ZOONAME_ENCLOSURENUMBER_INDIVIDUALNUMBER1+INDIVIDUALNUMBER2+...+INDIVIDUALNUMBERM
>
>*enclosurevideocode*: SPECIESNAME_ZOONAME_ENCLOSURENUMBER\*VIDEONUMBER1+VIDEONUMBER2+...+VIDEONUMBERN

While this seems highly specific on the first glance, the system might be tricked to be used for free-range obervations as well. A species could then for instance be a localisation identifier, there is only one (virtual) enclosure and one videostream. 

#### video storage
Furthermore, BOVIDS requires the videos (.avi format, 1fps) to be stored according to the following scheme in which the path including DATA_STORAGE is variable:
DATA_STORAGE/SPECIESNAME/ZOONAME/Videos/SPECIES_VIDEONUMBER/YYYY-MM-DD_SPECIESNAME_ZOONAME_VIDEONUMBER.avi

Notice that the bitrate of 1fps is necessary in order to make BOVIDS work in the current version. The program code would need to be modified at just a few places to make BOVIDS work on videos with higher framerate.

#### annotation storage
There are two types of annotations. It will be explained in a later section how the necessary files can be created by BOVIDS.

##### action classification storage
First, there are annotations of the nights (video annotations) which are assumed to be created with [BORIS](http://www.boris.unito.it/), Version *xyz* :left_speech_bubble: in which the *observation list* is exported as an xlsx-file called **boris-file**. Those boris-files need to be stored as follows where again, DATA_STORAGE is variable. (*Auswertung and Boris-Dateien are german expressions for evaluation and boris-datafile, this can be easily adjusted inside the code or just be used as dummy expressions*)

DATA_STORAGE/Auswertung/SPECIESNAME/ZOONAME/Auswertung/BORIS_KI/Boris-Dateien/YYYY-MM-DD_ENCLOSURECODE.xlsx

BOVIDS contains a script *action_classification/boris_to_csv.py* to convert those xlsx-files into machine readable .csv files (called **boris-csv-files**) per individual per night. Those boris-csv-files need to be stored in 

DATA_STORAGE/Auswertung/SPECIESNAME/ZOONAME/Auswertung/BORIS_KI/csv-Dateien/YYYY-MM-DD_ENCLOSURECODE_SUM-7s_pred.csv 

##### object detection storage
The second type of annotations stems from the annotation of single images for training the object detector. These are single images and corresponding label files created by labelImg in the following structure in which again ANNOTATION_STORAGE is variable (*Bilder is the german word for pictures/images and can be easily adapted in the code*).

Images: ANNOTATION_STORAGE/Bilder/SPECIESNAME/ZOONAME/ENCLOSURENUMBER/imagename.jpg

Corresponding Labels: ANNOTATION_STORAGE/Label/SPECIESNAME/ZOONAME/ENCLOSURENUMBER/imagename.xml


## Action classification
### Preparation
#### Convert video files
If the LUPUS system is used for recording, BOVIDS provides a handy converting option from the produced .asf-files ordered by channel into above's discussed structure using *action_classification/preparation/ConvertVideos.py*. If other recording systems are used, the .avi video files need to be created manually.  

#### Repair video files
Using LUPUS it happens quiete frequently that due to short drops in voltage a night consists of various parts or there are short sequences missing during a night. In order to make a realistic pose estimation over the whole observation time we need to make sure that all videos start and end at a known daytime and have a bitrate of 1fps. BOVIDS provides with *action_classification/preparation/video_processing.py* a collection of functions to concatenate multiple parts, to reduce the bitrate to 1fps and to fill in sequences of black frames into a video if some short sequences are missing. This program needs to be run in the video environment.

#### Annotate the video files with BORIS
In order to generate an initial training set, some nights need to be manually labelled and we propose to use BORIS (see action classification storage). It might be helpful to merge various video streams side to side in one video stream if many nights / enclosures / individuals shall be annotated at once. To this end, BOVIDS provides *action_classification/preparation/merge_video_files.py* which needs to be run in the video environment. Recall that the **boris-files** (.xlsx) need to be converted into **boris-csv-files** as described above. 

### Training of an initial network

### Offline hard example mining


## Object detection

### Preparation

### Training of an initial network

### Offline hard example mining


## Data prediction and evaluation

### Prediction

### Evaluation

### Presentation



## Acknowledgement
The yolov4 implementation of BOVIDS is based on the implementation by [taipingeric](https://github.com/taipingeric/yolo-v4-tf.keras). 






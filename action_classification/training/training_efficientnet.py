# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Max Hahn-Klimroth, Tobias Kapetanopoulos"
__copyright__ = "Copyright 2020, M. Hahn-Klimroth, T. Kapetanopoulos, J. Gübert, P. Dierkes"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "GPL-3.0"
__version__ = "1.0"

__status__ = "Development"


import pickle
import os

import tensorflow as tf
import imgaug.augmenters as iaa


#from keras.layers import GlobalAveragePooling2D, Dense, Dropout
#from keras.models import Sequential

from efficientnet.keras import EfficientNetB3
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import preprocess_input


# specify gpu to use ("0" or "1")
GPU_TO_USE = "1"
NUM_GPUS = 1

# Declare Error level and set precision
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_TO_USE
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('Used GPU:', GPU_TO_USE)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.config.set_soft_device_placement(True)
tf.compat.v1.logging.set_verbosity(0)
tf.autograph.set_verbosity(0)

# User configuration

BS_PER_GPU = 8
NUM_EPOCHS = 100
SAVE_EVERY_EPOCH = 1

HEIGHT = 300
WIDTH = 300


BEHAVIOR_LIST = ["standing", "lying"]
NUM_CLASSES = len(BEHAVIOR_LIST)


DATA_PATH = ''
VAL_PATH = ''

MODEL_WEIGHTS = ''

MODEL_SAVE_PATH_BASE = ''
CHECKPOINT_SAVE_PATH = ''
MODEL_NAME = ''


NUM_CHANNELS = 3
INPUT_SHAPE = (WIDTH, HEIGHT, NUM_CHANNELS)




def augment_data(img):
    
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes(iaa.Crop(px=(4, 16))),   
        # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), 
        # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)),
        # blur images with a sigma of 0 to 3.0
        iaa.Multiply((0.7, 1.4), per_channel=0.0),
        iaa.LinearContrast((0.75, 1.25)),
        iaa.Affine(rotate=(-25, 25), shear=(-8, 8))
        ], random_order = True)
    
    
    seq_det = seq.to_deterministic()
    aug_image = seq_det.augment_image(img)

    return preprocess_input(aug_image)
    
  

def get_steps(data_folder = DATA_PATH, val_folder = VAL_PATH, 
              classes = BEHAVIOR_LIST, gpus = NUM_GPUS, bs = BS_PER_GPU):
    
    base, val = 0, 0
    for j in range(len(classes)):
        base += len(os.listdir(data_folder + str(j)))
        val += len(os.listdir(val_folder + str(j)))
    
    steps_per_epoch = int( 1.0*base / float(bs*gpus) )
    val_steps = int( 1.0*val / float(bs*gpus) )
    
    return steps_per_epoch, val_steps
   

def prepare_datasets(input_path = DATA_PATH, val_path = VAL_PATH, 
                     batch_size = BS_PER_GPU, size = (HEIGHT, WIDTH)):
    
    train_datagen = ImageDataGenerator(
             rescale=1./255,
             preprocessing_function=augment_data,
             validation_split=0)

    
    train_generator = train_datagen.flow_from_directory(
        input_path,
        shuffle = True,
        batch_size= batch_size,
        class_mode='categorical',
        target_size = size)
    
    validation_datagen = ImageDataGenerator(rescale=1./255,
                                            preprocessing_function=preprocess_input)
    
    validation_generator = validation_datagen.flow_from_directory(
        val_path,
        shuffle=False,
        class_mode='categorical',
        target_size= size)
    
    return train_generator, validation_generator



def train_model(train_generator, validation_generator, save_path, 
                checkpoint_path, model_name, model_weights = MODEL_WEIGHTS,
                classes = BEHAVIOR_LIST):
    
    def build_model(num_classes, model_weights = model_weights):
        
        if model_weights == 'imagenet':               
            base_model = EfficientNetB3(weights='imagenet', include_top=False)                 
            
               
            model = Sequential([
                base_model,
                layers.GlobalAveragePooling2D(name="gap"),
                layers.Dropout(0.2, name="dropout_out"),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            opt = tf.keras.optimizers.Adam()
            model.compile(
                      optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        elif model_weights.endswith('.h5'):
            model = tf.keras.models.load_model(model_weights)
            optimizer = tf.keras.optimizers.Adam()
            model.compile(
                optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
            )

        else:
            print("Invalid weights.")
            return
    

        return model   
    
     
    
    if not os.path.exists(save_path+"logs/fit/"):
        os.makedirs(save_path+"logs/fit/")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path+model_name+"_cp-{epoch:04d}.h5",
                                                     verbose=1, 
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     monitor = 'val_accuracy',
                                                     mode = 'max',
                                                     period=SAVE_EVERY_EPOCH)
    
    model = build_model(len(classes))
    steps_per_epoch, val_steps = get_steps()
    history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=NUM_EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=val_steps,
                              callbacks=[cp_callback],
                              initial_epoch = 0
                             )

    
    
    model.save(save_path+model_name+'.h5')
    
    with open(save_path+model_name+"_history.txt", "wb") as fp:
        pickle.dump(history.history, fp)




model_name_resnet = MODEL_NAME
resnet_savepath = MODEL_SAVE_PATH_BASE + model_name_resnet + '/'
checkpoint_path_resnet = CHECKPOINT_SAVE_PATH + model_name_resnet + '_checkpoint/'

train_generator, validation_generator = prepare_datasets(input_path = DATA_PATH, 
                                                         val_path = VAL_PATH)

print("Created datasets... Starting with training.")
train_model(train_generator, validation_generator,
            save_path = resnet_savepath, 
            checkpoint_path = checkpoint_path_resnet, 
            model_name = model_name_resnet)



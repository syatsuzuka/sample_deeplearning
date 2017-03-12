# coding: utf-8

"""
A sample code for deep learning

Copyright (C) 2017 Shunjiro Yatsuzuka
"""

__author__ = "Shunjiro Yatsuzuka"
__copyright__ = "Copyright (c) 2017 Shunjiro Yatsuzuka"
__date__ = "March.12, 2017"
__version__ = "0.1"


#======= Import Modules =======

#----- General -----

import os
import sys
import time
import importlib

#----- Keras -----

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import numpy as np

#----- SDL -----

from common import Log


#======= ClassifyImage Class =======

class ClassifyImage():

    """
    Class to classify images
    """


    #======= Load Data =======

    def load_data(self):

        """
        Function to load datasets from cifer10
        """

        #======= Start Message =======

        self.log.output_msg(1, 1, "ClassifyImage.load_data() started")


        #======= Download Data =======

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()


        #======= Convert training data into binary style =======

        self.y_train = np_utils.to_categorical(self.y_train, self.class_num)
        self.y_test = np_utils.to_categorical(self.y_test, self.class_num)

        self.log.output_msg(1, 1, "x_train.shape = {0}".format(self.x_train.shape))
        self.log.output_msg(1, 1, "y_train.shape = {0}".format(self.y_train.shape))
        self.log.output_msg(1, 1, "x_train.shape[0] = {0}".format(self.x_train.shape[0]))
        self.log.output_msg(1, 1, "y_train.shape[0] = {0}".format(self.y_train.shape[0]))
        self.log.output_msg(1, 1, "x_test.shape[0] = {0}".format(self.x_test.shape[0]))
        self.log.output_msg(1, 1, "y_test.shape[0] = {0}".format(self.y_test.shape[0]))


        #======= End Message =======

        self.log.output_msg(1, 1, "ClassifyImage.load_data() ended")


    #======= Preprocessing =======

    def preproc(self):


        #======= Start Message =======

        self.log.output_msg(1, 1, "ClassifyImage.preproc() started")


        self.log.output_msg(1, 1, "x_train[0][0][0] = {0}".format(self.x_train[0][0][0]))
        self.log.output_msg(1, 1, "x_test[0][0][0] = {0}".format(self.x_test[0][0][0]))

        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        self.log.output_msg(1, 1, "x_train[0][0][0] = {0}".format(self.x_train[0][0][0]))
        self.log.output_msg(1, 1, "x_test[0][0][0] = {0}".format(self.x_test[0][0][0]))


        #======= End Message =======

        self.log.output_msg(1, 1, "ClassifyImage.preproc() ended")


    #======= Classification =======

    def run(self):


        #======= Start Message =======

        self.log.output_msg(1, 1, "ClassifyImage.run() started")


        #======= Create Network Model =======

        model_module = importlib.import_module(self.model_path)
        input_shape = self.x_train.shape[1:]
        output_shape = self.class_num

        self.log.output_msg(1, 1, "input_shape = {0}".format(input_shape))
        self.log.output_msg(1, 1, "output_shape = {0}".format(output_shape))

        self.model = model_module.create_model(
            input_shape, 
            output_shape
        )


        #======= Define Optimization Model =======

        sgd = SGD(
            lr=self.sgd_lr, 
            decay=self.sgd_decay, 
            momentum=self.sgd_momentum, 
            nesterov=True
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy']
        )


        self.model.summary()


        #======= Start Training / Validation =======

        if ( self.data_aug ):
        
            self.log.output_msg(1, 1, "Using Data Augmentation")

            #======= Data Augmentation =======

            dataaug = ImageDataGenerator(
                featurewise_center=self.data_aug_fcenter,
                samplewise_center=self.data_aug_scenter,
                featurewise_std_normalization=self.data_aug_fstdnorm,
                samplewise_std_normalization=self.data_aug_sstdnorm,
                zca_whitening=self.data_aug_zca,
                rotation_range=self.data_aug_rot_range,
                width_shift_range=self.data_aug_wshift_range,
                height_shift_range=self.data_aug_hshift_range,
                horizontal_flip=self.data_aug_hflip,
                vertical_flip=self.data_aug_vflip
            )


            #======= Calc Weight Matrix =======

            dataaug.fit(self.x_train)
            self.model.fit_generator(
                dataaug.flow(
                    self.x_train, 
                    self.y_train,
                    batch_size=self.batch_size
                ),
                samples_per_epoch=self.x_train.shape[0],
                nb_epoch=self.epoch_num,
                validation_data=(
                    self.x_test,
                    self.y_test
                )
            )
    

        else:
        
            self.log.output_msg(1, 1, "Not using Data Augmentation")

            #======= Calc Weight Matrix =======

            self.model.fit(
                self.x_train,
                self.y_train,
                batch_size=self.batch_size,
                nb_epoch=self.epoch_num,
                validation_data=(self.x_test, self.y_test),
                shuffle=True,
                callbacks=[TensorBoard(log_dir=self.log_dir)]
            )


        #======= End Message =======

        self.log.output_msg(1, 1, "ClassifyImage.run() ended")
            

    #======= Initialization =======

    def __init__(self):


        """
        Constructor
        """

        #======= Setup Log =======

        self.log = Log(os.environ['SDL_LOG_LEVEL'])


        #======= Start Message =======

        self.log.output_msg(1, 1, "ClassifyImage.__init__() started")


        #======= Setup Log Directory =======

        if 'SDL_LOG_DIR' in os.environ:
            self.log_dir = os.environ['SDL_LOG_DIR']
        else:
            self.log_dir = './log'

        self.log.output_msg(1, 1, "self.log_dir = {0}".format(self.log_dir))


        #======= Setup Model Path =======

        if 'SDL_MODEL_PATH' in os.environ:
            self.model_path = os.environ['SDL_MODEL_PATH']
        else:
            self.model_path = "model.sample_cnn"

        self.log.output_msg(1, 1, "self.model_path = {0}".format(self.model_path))


        #======= Setup Class Num =======

        if 'SDL_CLASS_NUM' in os.environ:
            self.class_num = int(os.environ['SDL_CLASS_NUM'])
        else:
            self.class_num = 10

        self.log.output_msg(1, 1, "self.class_num = {0}".format(self.class_num))


        #======= Setup Row Num =======

        if 'SDL_ROW_NUM' in os.environ:
            self.row_num = int(os.environ['SDL_ROW_NUM'])
        else:
            self.row_num = 32

        self.log.output_msg(1, 1, "self.row_num = {0}".format(self.row_num))

        #======= Setup Col Num =======

        if 'SDL_COL_NUM' in os.environ:
            self.col_num = int(os.environ['SDL_COL_NUM'])
        else:
            self.col_num = 32

        self.log.output_msg(1, 1, "self.col_num = {0}".format(self.col_num))

        #======= Set Channel Num =======

        if 'SDL_CHANNEL_NUM' in os.environ:
            self.channel_num = int(os.environ['SDL_CHANNEL_NUM'])
        else:
            self.channel_num = 3

        self.log.output_msg(1, 1, "self.channel_num = {0}".format(self.channel_num))
        
        #======= Setup Batch Size =======

        if 'SDL_BATCH_SIZE' in os.environ:
            self.batch_size = int(os.environ['SDL_BATCH_SIZE'])
        else:
            self.batch_size = 32

        self.log.output_msg(1, 1, "self.batch_siize = {0}".format(self.batch_size))

        #======= Setup Epoch Number =======

        if 'SDL_EPOCH_NUM' in os.environ:
            self.epoch_num = int(os.environ['SDL_EPOCH_NUM'])
        else:
            self.epoch_num = 32

        self.log.output_msg(1, 1, "self.epoch_num = {0}".format(self.epoch_num))

        #======= Setup SGD Learning Rate =======

        if 'SDL_SGD_LR' in os.environ:
            self.sgd_lr = float(os.environ['SDL_SGD_LR'])
        else:
            self.sgd_lr = 0.01

        self.log.output_msg(1, 1, "self.sgd_lr = {0}".format(self.sgd_lr))


        #======= Setup SGD Decay =======

        if 'SDL_SGD_Decay' in os.environ:
            self.sgd_decay = float(os.environ['SDL_SGD_DECAY'])
        else:
            self.sgd_decay = 1e-4

        self.log.output_msg(1, 1, "self.sgd_decay = {0}".format(self.sgd_decay))


        #======= Setup SGD Momentum =======

        if 'SDL_SGD_MOMENTUM' in os.environ:
            self.sgd_momentum = float(os.environ['SDL_SGD_MOMENTUM'])
        else:
            self.sgd_momentum = 0.9

        self.log.output_msg(1, 1, "self.sgd_momentum = {0}".format(self.sgd_momentum))


        #======= Setup Data Augmentation =======

        if 'SDL_DATA_AUG' in os.environ:
            self.data_aug = bool(os.environ['SDL_DATA_AUG'])
        else:
            self.data_aug = True

        self.log.output_msg(1, 1, "self.data_aug = {0}".format(self.data_aug))


        #======= Setup Featurewise Center =======

        if 'SDL_DATA_AUG_FCENTER' in os.environ:
            self.data_aug_fcenter = bool(os.environ['SDL_DATA_AUG_FCENTER'])
        else:
            self.data_aug_fcenter = False

        self.log.output_msg(1, 1, "self.data_aug_fcenter = {0}".format(self.data_aug_fcenter))

        #======= Setup Samplewise Center =======

        if 'SDL_DATA_AUG_SCENTER' in os.environ:
            self.data_aug_scenter = bool(os.environ['SDL_DATA_AUG_SCENTER'])
        else:
            self.data_aug_scenter = False

        self.log.output_msg(1, 1, "self.data_aug_scenter = {0}".format(self.data_aug_scenter))

        #======= Setup Featurewise STD Normalization =======

        if 'SDL_DATA_AUG_FSTDNORM' in os.environ:
            self.data_aug_fstdnorm = bool(os.environ['SDL_DATA_AUG_FSTDNORM'])
        else:
            self.data_aug_fstdnorm = False

        self.log.output_msg(1, 1, "self.data_aug_fstdnorm = {0}".format(self.data_aug_fstdnorm))

        #======= Setup Samplewise STD Normalization =======

        if 'SDL_DATA_AUG_SSTDNORM' in os.environ:
            self.data_aug_sstdnorm = bool(os.environ['SDL_DATA_AUG_SSTDNORM'])
        else:
            self.data_aug_sstdnorm = False

        self.log.output_msg(1, 1, "self.data_aug_sstdnorm = {0}".format(self.data_aug_sstdnorm))


        #======= Setup ZCA Whitening =======

        if 'SDL_DATA_AUG_ZCA' in os.environ:
            self.data_aug_zca = bool(os.environ['SDL_DATA_AUG_ZCA'])
        else:
            self.data_aug_zca = False

        self.log.output_msg(1, 1, "self.data_aug_zca = {0}".format(self.data_aug_zca))

        #======= Setup Rot Range =======

        if 'SDL_DATA_AUG_ROT_RANGE' in os.environ:
            self.data_aug_rot_range = int(os.environ['SDL_DATA_AUG_ROT_RANGE'])
        else:
            self.data_aug_rot_range = 0

        self.log.output_msg(1, 1, "self.data_aug_rot_range = {0}".format(self.data_aug_rot_range))

        #======= Setup Width Shift Range =======

        if 'SDL_DATA_AUG_WSHIFT_RANGE' in os.environ:
            self.data_aug_wshift_range = int(os.environ['SDL_DATA_AUG_WSHIFT_RANGE'])
        else:
            self.data_aug_wshift_range = 0

        self.log.output_msg(1, 1, "self.data_aug_wshift_range = {0}".format(self.data_aug_wshift_range))

        #======= Setup Height Shift Range =======

        if 'SDL_DATA_AUG_HSHIFT_RANGE' in os.environ:
            self.data_aug_hshift_range = int(os.environ['SDL_DATA_AUG_HSHIFT_RANGE'])
        else:
            self.data_aug_hshift_range = 0

        self.log.output_msg(1, 1, "self.data_aug_hshift_range = {0}".format(self.data_aug_hshift_range))

        #======= Setup Horizontal Flip =======

        if 'SDL_DATA_AUG_HFLIP' in os.environ:
            self.data_aug_hflip = bool(os.environ['SDL_DATA_AUG_HFLIP'])
        else:
            self.data_aug_hflip = False

        self.log.output_msg(1, 1, "self.data_aug_hflip = {0}".format(self.data_aug_hflip))

        #======= Setup Vertical Flip =======

        if 'SDL_DATA_AUG_VFLIP' in os.environ:
            self.data_aug_vflip = bool(os.environ['SDL_DATA_AUG_VFLIP'])
        else:
            self.data_aug_vflip = False

        self.log.output_msg(1, 1, "self.data_aug_vflip = {0}".format(self.data_aug_vflip))



        #======= End Message =======

        self.log.output_msg(1, 1, "ClassifyImage.__init__() ended")

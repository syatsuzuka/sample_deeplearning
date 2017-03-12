#!/bin/bash

#========================================================================
# FILE NAME:    classify_image.param
# FUNCTION:     Parameter file for classify_image.py
# VERSION:      1.0
# AUTHOR:       S.Yatsuzuka
#
# Copyright (C) 2017 Shunjiro Yatsuzuka
#========================================================================

export SDL_CMD="python ${SDL_HOME}/src/main.py"
export SDL_LOG_LEVEL="1"
export SQL_LOG_DIR="${SDL_HOME}/log"
export SDL_SRC_DIR="${SDL_HOME}/src"
export SDL_MODEL_PATH="model.sample_cnn2"
export SDL_CLASS_NUM="10"
export SDL_ROW_NUM="32"
export SDL_COL_NUM="32"
export SDL_CHANNEL_NUM="3"
export SDL_BATCH_SIZE="32"
export SDL_EPOCH_NUM="30"
export SDL_SGD_LR="0.01"
export SDL_SGD_DECAY="1e-4"
export SDL_SGD_MOMENTUM="0.9"
export SDL_DATA_AUG="True"
export SDL_DATA_AUG_FCENTER=""
export SDL_DATA_AUG_SCENTER=""
export SDL_DATA_AUG_FSTDNORM=""
export SDL_DATA_AUG_SSTDNORM=""
export SDL_DATA_AUG_ZCA=""
export SDL_DATA_AUG_ROT_RANGE="0"
export SDL_DATA_AUG_WSHIFT_RANGE="0"
export SDL_DATA_AUG_HSHIFT_RANGE="0"
export SDL_DATA_AUG_HFLIP="True"
export SDL_DATA_AUG_VFLIP=""


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

#----- SDL -----
from common import Log
from classify_image import ClassifyImage


#======= Global Variable =======

log = Log(os.environ['SDL_LOG_LEVEL'])


#======= Main Function =======

if __name__ == '__main__':

	#======= Start Message =======

	log.output_msg(1, 1, "main() started")
	log.output_msg(1, 1, "log.log_level = {0}".format(log.log_level))

	#======= Start Process =======

	app = ClassifyImage()
	app.load_data()
	app.preproc()
	app.run()

	#======= End Message =======

	log.output_msg(1, 1, "main() ended")


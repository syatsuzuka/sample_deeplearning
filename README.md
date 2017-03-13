# Sample Code for deep learning

This is a sample code for deep learning with Keras + Tensorflow.


[Accuracy of model]
loss: 0.2392 - acc: 0.9144 - val_loss: 0.6078 - val_acc: 0.8145


[Prerequisite]

* Install Python 3.x and Keras

It is recommended to use Anaconda to build work environment.


[How To Use]

Step.1) Get source code.

 $ git clone https://github.com/syatsuzuka/sample_deeplearning.git

Step.2) Setup necessary environment variable in .bashrc.

PARAMETER|DESCRIPTION|EXAMPLE
---------|-----------|-------
SDL_HOME|top directory of the code|${HOME}/sampel_deeplearning
SDL_LOG_LEVEL|Log Level|"1"


Step.3) Execute the following command.

 $ ./bin/run.sh param/classify_image.param


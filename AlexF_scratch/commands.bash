#!/bin/bash

# uzip the training data into /train folder
#unzip train.zip

# convert train/*.tif and *_maks.tif data to numpy arrays 
# using code from 
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py

python -c 'import BP_utils; \
BP_utils.create_train_data(); \
BP_utils.create_test_data()'



 

#!/bin/bash

## the kaggle data files must have already been unzipped into the 
## train/ and test/ folders 

# convert train/*.tif and *_maks.tif data to numpy arrays 
# using code from 
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/data.py

python -c 'import BP_utils; \
BP_utils.create_train_data(); \
BP_utils.create_test_data()'



 

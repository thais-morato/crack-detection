"""



"""
import os
from base.constants import *
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16

def preprocess():

    trainDirectory = os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN)
    testDirectory = os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN)

    trainDatagen = ImageDataGenerator(
        preprocessing_function=vgg16.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    testDatagen = ImageDataGenerator(
        preprocessing_function=vgg16.preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    trainSet = trainDatagen.flow_from_directory(trainDirectory, target_size = TARGET_IMAGE_SIZE, batch_size = 32, class_mode = 'categorical')
    testSet = testDatagen.flow_from_directory(testDirectory, target_size = TARGET_IMAGE_SIZE, batch_size = 32, class_mode = 'categorical')

    return (trainSet, testSet)


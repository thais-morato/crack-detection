"""

    TODO: Documentation

"""

import os
import base.params as params
import base.constants as consts
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

def _getDirectory(path, set):
    if set == consts.SetEnum.train:
        return _getSetDirectory(path, consts.AP_FOLDER_TRAIN)
    elif set == consts.SetEnum.validation:
        return _getSetDirectory(path, consts.AP_FOLDER_VALIDATION)
    else:
        return _getSetDirectory(path, consts.AP_FOLDER_TEST)

def _getSetDirectory(path, setFolder):
    return os.path.join(path, setFolder)

def _getDataset(directory, preprocessingFunction, applyDataAugmentation):
    dataGenerator = _getDataGenerator(preprocessingFunction, applyDataAugmentation)
    
    return dataGenerator.flow_from_directory(
        directory=directory,
        target_size=consts.TARGET_IMAGE_SIZE,
        class_mode='binary',
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        seed=8)

def _getDataGenerator(preprocessingFunction, applyDataAugmentation):
    # Generate batches of tensor image data
    # Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    if applyDataAugmentation:
        return _getDataGeneratorWithAugmentation(preprocessingFunction)
    else:
        return _getDataGeneratorWithoutAugmentation(preprocessingFunction)

def _getDataGeneratorWithoutAugmentation(preprocessingFunction):
    return ImageDataGenerator(preprocessing_function=preprocessingFunction)

def _getDataGeneratorWithAugmentation(preprocessingFunction):
    return ImageDataGenerator(
        preprocessing_function=preprocessingFunction,
        horizontal_flip=True,
        vertical_flip=True)

def getPreprocessedDataset(path, set, applyDataAugmentation=False):
    directory = _getDirectory(path, set)
    dataset = _getDataset(directory, preprocess_input, applyDataAugmentation)
    return dataset

def getRawDataset(path, set, applyDataAugmentation=False):
    directory = _getDirectory(path, set)
    dataset = _getDataset(directory, None, applyDataAugmentation)
    return dataset
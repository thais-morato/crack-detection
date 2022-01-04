import numpy as np
import os, shutil, sys
from keras import layers
import base.params as params
from keras.models import Model
import base.constants as consts
from keras.applications import vgg16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
from keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model, save_model

def _getBaseModel():
    # Instanciating a base model pre-trained on ImageNet dataset
    baseModel = vgg16.VGG16(
        input_shape=consts.TARGET_IMAGE_SIZE + (3,),
        weights="imagenet",
        include_top=False)

    # Freezing the weigths from the base model
    baseModel.trainable = False

    return baseModel

def _addClassificationLayers(modelName, baseModel):
    flatten = layers.Flatten(name='flatten')(baseModel.output)
    hidden1 = layers.Dense(64, activation='relu', name='hidden_1')(flatten)
    hidden2 = layers.Dense(32, activation='relu', name='hidden_2')(hidden1)
    prediction = layers.Dense(1, activation=sigmoid, name='prediction')(hidden2)
    return Model(inputs=baseModel.input, outputs=prediction, name=modelName)

def _getOptimizer():
    return Adam(learning_rate=params.LEARNING_RATE)

def _compileModel(model):
    optimizer = _getOptimizer()
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

def _createFolderStructure(modelName, fromScratch, isFineTuning):   
    if fromScratch and os.path.isdir(consts.AP_FOLDER_CHECKPOINTS):
        shutil.rmtree(consts.AP_FOLDER_CHECKPOINTS)
    
    for path in [
        consts.AP_FOLDER_CHECKPOINTS,
        consts.AP_FOLDER_LOG,
        consts.AP_FOLDER_MODELS]:
        if not os.path.isdir(path):
            os.mkdir(path)

    if isFineTuning:
        logPath = os.path.join(consts.AP_FOLDER_LOG, modelName + consts.FINE_TUNING_COMPLEMENT + consts.EXTENSION_LOG)
    else:
        logPath = os.path.join(consts.AP_FOLDER_LOG, modelName + consts.EXTENSION_LOG)
    
    if fromScratch and os.path.isfile(logPath):
        os.remove(logPath)

def _getModelPath(modelName, isFineTuning):
    if isFineTuning:
        path = os.path.join(consts.AP_FOLDER_MODELS, modelName + consts.FINE_TUNING_COMPLEMENT + consts.EXTENSION_MODEL)
    else:
        path = os.path.join(consts.AP_FOLDER_MODELS, modelName + consts.EXTENSION_MODEL)
    return path

def _getLastCheckpointPath():
    checkpoints = os.listdir(consts.AP_FOLDER_CHECKPOINTS)
    if len(checkpoints) == 0:
        sys.exit("No model checkpoints were found")
    path = os.path.join(consts.AP_FOLDER_CHECKPOINTS, checkpoints[-1])
    input("Checkpoint found: " + checkpoints[-1] + ". Press enter to resume training")
    return path

def _getInitialEpoch(path):
    return (int)(path.split("-")[-1].split(consts.EXTENSION_MODEL)[0])

def _getCallbacks(modelName, isFineTuning):
    if isFineTuning:
        filePath = os.path.join(consts.AP_FOLDER_CHECKPOINTS, modelName + consts.FINE_TUNING_COMPLEMENT + consts.EXTENSION_CHECKPOINT)
    else:
        filePath = os.path.join(consts.AP_FOLDER_CHECKPOINTS, modelName + consts.EXTENSION_CHECKPOINT)
    
    checkpoint = ModelCheckpoint(
        filepath=filePath,
        save_freq='epoch',
        verbose=1)

    if isFineTuning:
        filePath = os.path.join(consts.AP_FOLDER_LOG, modelName + consts.FINE_TUNING_COMPLEMENT + consts.EXTENSION_LOG)
    else:
        filePath = os.path.join(consts.AP_FOLDER_LOG, modelName + consts.EXTENSION_LOG)
    
    log = CSVLogger(filePath, append=True)
    return [checkpoint, log]

def _trainModel(modelName, model, trainSet, validationSet, callbacks, isFineTuning, initialEpoch=0):
    modelHistory = model.fit(
        trainSet,
        validation_data=validationSet,
        epochs=params.EPOCHS,
        shuffle=True,
        steps_per_epoch=np.ceil(trainSet.samples/params.BATCH_SIZE),
        validation_steps=np.ceil(validationSet.samples/params.BATCH_SIZE),
        initial_epoch=initialEpoch,
        callbacks=callbacks,
        verbose=2)
    path = _getModelPath(modelName, isFineTuning=isFineTuning)
    save_model(model, path)
    return modelHistory

def _buildModel(modelName):
    model = _getBaseModel()
    model = _addClassificationLayers(modelName, model)
    return model

def _unfreezeModel(model):
    model.trainable = True

def trainModelFromScratch(modelName, trainSet, validationSet):
    _createFolderStructure(modelName, fromScratch=True, isFineTuning=False)
    model = _buildModel(modelName)
    _compileModel(model)
    callbacks = _getCallbacks(modelName, isFineTuning=False)
    modelHistory = _trainModel(modelName, model, trainSet, validationSet, callbacks, isFineTuning=False)
    return modelHistory

def resumeTraining(modelName, trainSet, validationSet):
    _createFolderStructure(modelName, fromScratch=False, isFineTuning=False)
    path = _getLastCheckpointPath()
    model = load_model(path)
    _compileModel(model)
    callbacks = _getCallbacks(modelName, isFineTuning=False)
    initialEpoch = _getInitialEpoch(path)
    modelHistory = _trainModel(modelName, model, trainSet, validationSet, callbacks, isFineTuning=False, initialEpoch=initialEpoch)
    return modelHistory

def fineTuneModelFromScratch(modelName, trainSet, validationSet):
    _createFolderStructure(modelName, fromScratch=True, isFineTuning=True)
    originalModelPath = _getModelPath(modelName, isFineTuning=False)
    model = load_model(originalModelPath)
    _unfreezeModel(model)
    _compileModel(model)
    callbacks = _getCallbacks(isFineTuning=True)
    modelHistory = _trainModel(modelName, model, trainSet, validationSet, callbacks, isFineTuning=True)
    return modelHistory

def resumeFineTuning(modelName, trainSet, validationSet):
    _createFolderStructure(modelName, fromScratch=False, isFineTuning=True)
    path = _getLastCheckpointPath()
    model = load_model(path)
    _compileModel(model)
    callbacks = _getCallbacks(isFineTuning=True)
    initialEpoch = _getInitialEpoch(path)
    modelHistory = _trainModel(modelName, model, trainSet, validationSet, callbacks, isFineTuning=True, initialEpoch=initialEpoch)
    return modelHistory
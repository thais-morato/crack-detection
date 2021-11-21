import os
from keras.applications import vgg16
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from base.params import *
from base.constants import *
from datetime import datetime
from keras.callbacks import ModelCheckpoint

NUMBER_OF_CLASSES = 2

def train(trainSet, testSet):
    vggModel = vgg16.VGG16(input_shape = IMAGE_SIZE + [3], weights = "imagenet", include_top = False)

    for layer in vggModel.layers:
        layer.trainable = False
    
    x = Flatten()(vggModel.output)
    prediction = Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
    model = Model(inputs = vggModel.input, outputs = prediction)
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'rmsprop',
        metrics = ['accuracy']
    )

    filePath = os.path.join(AP_FOLDER_MODELS, MODEL_NAME)
    checkpoint = ModelCheckpoint(filepath = filePath, verbose = 2, save_best_only = True)
    callbacks = [checkpoint]

    trainingStart = datetime.now()

    # TODO: change to model.fit
    modelHistory = model.fit(
        trainSet,
        validation_data = testSet,
        epochs = 10,
        steps_per_epoch = 5,
        validation_steps = 32,
        callbacks = callbacks,
        verbose = 2
    )

    trainingDuration = datetime.now() - trainingStart

    return (modelHistory, trainingDuration)

def createDestinationPath():
    path = AP_FOLDER_MODELS
    if not os.path.isdir(path):
        os.mkdir(path)
"""

    Run this file to evaluate a CNN Model on the test set. The outputs are the
    confusion matrix, accuracy and examples of true positive, false positive,
    true negative and false negative samples.

"""

import os, sys
import pandas as pd
import seaborn as sn
import base.params as params
import matplotlib.pyplot as plt
import base.constants as consts
import preprocessing.preprocessing as prep
from tensorflow.keras.models import load_model

class _RawAndPreprocessedImage:
    raw = None
    prep = None
    def __init__(self, raw, prep):
        self.raw = raw
        self.prep = prep

def _getModelName():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return None

def _loadModel(modelName):
    if modelName != None:
        modelFileName = modelName
    elif params.IS_FINE_TUNING:
        modelFileName = params.MODEL_NAME + consts.FINE_TUNING_COMPLEMENT
    else:
        modelFileName = params.MODEL_NAME
    modelFile = os.path.join(consts.AP_FOLDER_MODELS, modelFileName + consts.EXTENSION_MODEL)
    print("Model: " + modelFileName + consts.EXTENSION_MODEL)
    return load_model(modelFile)

def _getRawAndPreprocessedTestSets():
    rawTestSet = prep.getRawDataset(set=consts.SetEnum.test, applyDataAugmentation=False)
    if(params.APPLY_PREPROCESSING):
        testSet = prep.getPreprocessedDataset(set=consts.SetEnum.test, applyDataAugmentation=False)
    else:
        testSet = prep.getRawDataset(set=consts.SetEnum.test, applyDataAugmentation=False)
    return (rawTestSet, testSet)

def _getReferenceValues(dataset):
    positiveValue = dataset.class_indices[consts.AP_FOLDER_ANOMALOUS]
    negativeValue = dataset.class_indices[consts.AP_FOLDER_NORMAL]
    return (positiveValue, negativeValue)

def _getConfusionMatrixArrays(model, testSet, rawTestSet, positiveValue):
    truePositives = []
    falsePositives = []
    trueNegatives = []
    falseNegatives = []

    imgCount = 0
    rawTestSetIter = iter(rawTestSet)
    for images, labels in testSet:
        rawImages, rawLabels = next(rawTestSetIter)
        predictions = model.predict(images)
        for i in range(len(predictions)):
            if int(labels[i]) == positiveValue:
                if int(predictions[i]) == positiveValue:
                    truePositives.append(_RawAndPreprocessedImage(rawImages[i], images[i]))
                else:
                    falseNegatives.append(_RawAndPreprocessedImage(rawImages[i], images[i]))
            else:
                if int(predictions[i]) == positiveValue:
                    falsePositives.append(_RawAndPreprocessedImage(rawImages[i], images[i]))
                else:
                    trueNegatives.append(_RawAndPreprocessedImage(rawImages[i], images[i]))
            imgCount += 1
            if imgCount == testSet.samples:
                return (truePositives, falsePositives, trueNegatives, falseNegatives)

def _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives):
    array = [[len(truePositives), len(falsePositives)],
            [len(falseNegatives), len(trueNegatives)]]
    df_cm = pd.DataFrame(
                array,
                index=pd.Index([params.FOLDER_ANOMALOUS, params.FOLDER_NORMAL], name="Predicted"),
                columns= pd.Index([params.FOLDER_ANOMALOUS, params.FOLDER_NORMAL], name="Actual"))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")

def _printAccuracy(truePositives, falsePositives, trueNegatives, falseNegatives):
    correctPredictionAmount = len(truePositives) + len(trueNegatives)
    incorrectPredictionAmount = len(falsePositives) + len(falseNegatives)
    accuracy = correctPredictionAmount / (correctPredictionAmount + incorrectPredictionAmount) * 100
    print("Accuracy: %.2f%%" % accuracy)

def _plotExamples(title, rawAndPreprocessedImages):
    fig = plt.figure(title, figsize=(7.8, 1.5))
    amount = min(5, len(rawAndPreprocessedImages))
    plt.subplot(1, 5, 1)
    plt.axis("off")
    for i in range(amount):
        plt.subplot(1, 5, i+1)
        plt.imshow(rawAndPreprocessedImages[i].raw.astype('uint8'))
        plt.axis("off")
    plt.subplots_adjust(bottom=0, top=0.8)
    plt.suptitle(title)
    fig.show()

def run():
    modelName = _getModelName()
    model = _loadModel(modelName)
    (rawTestSet, testSet) = _getRawAndPreprocessedTestSets()
    (positiveValue, negativeValue) = _getReferenceValues(testSet)
    (truePositives,
        falsePositives,
        trueNegatives,
        falseNegatives) = _getConfusionMatrixArrays(model, testSet, rawTestSet, positiveValue)
    _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives)
    _printAccuracy(truePositives, falsePositives, trueNegatives, falseNegatives)
    _plotExamples("True Positives", truePositives)
    _plotExamples("False Positives", falsePositives)
    _plotExamples("True Negatives", trueNegatives)
    _plotExamples("False Negatives", falseNegatives)
    plt.show()

if __name__ == "__main__":
    run()


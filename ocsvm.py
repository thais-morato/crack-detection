import os
import numpy as np
import pandas as pd
import seaborn as sn
import base.params as params
import matplotlib.pyplot as plt
import base.constants as consts
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

def _getFilePaths(subset, isAnomalous):
    if subset == consts.SetEnum.test:
        subsetFolder = consts.AP_FOLDER_TRAIN
    elif subset == consts.SetEnum.validation:
        subsetFolder = consts.AP_FOLDER_VALIDATION
    else:
        subsetFolder = consts.AP_FOLDER_TEST
    
    if isAnomalous:
        classFolder = consts.AP_FOLDER_ANOMALOUS
    else:
        classFolder = consts.AP_FOLDER_NORMAL
    
    folder = os.path.join(consts.AP_PATH_DATASET,
                        subsetFolder,
                        classFolder)
    return [os.path.join(folder, img) for img in os.listdir(folder)]

def _getPca(x):
    pca = PCA(n_components=params.PCA_COMPONENTS)
    pca.fit(x)
    return pca

def _getTrainSamples():
    normalFiles = _getFilePaths(consts.SetEnum.train, False)
    x = []
    for file in normalFiles:
        img = plt.imread(file)
        img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2])
        x.append(img)
    return x

def _trainOcSvm(x, pca):
    transformedX = pca.transform(x)
    ocSvm = OneClassSVM(kernel="sigmoid", gamma="auto")
    ocSvm.fit(transformedX)
    return ocSvm

def _getTestSamples():
    anomalousFiles = _getFilePaths(consts.SetEnum.test, True)
    normalFiles = _getFilePaths(consts.SetEnum.test, False)
    x = []
    y = []
    for file in normalFiles:
        img = plt.imread(file)
        img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2])
        x.append(img)
        y.append(1)
    for file in anomalousFiles:
        img = plt.imread(file)
        img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2])
        x.append(img)
        y.append(-1)
    return x, y

def _predict(x, pca, ocSvm):
    predictions = []
    nBatches = int(np.ceil(len(x)/params.BATCH_SIZE))
    for i in range(nBatches):
        initial = i*params.BATCH_SIZE
        final = min(initial + params.BATCH_SIZE, len(x))
        xBatch = x[initial:final]
        transformedX = pca.transform(xBatch)
        predictionsBatch = ocSvm.predict(transformedX)
        predictions.extend(predictionsBatch)
    return predictions

def _evaluate(y, predictions):
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0

    for i in range(len(y)):
        if y[i] == -1:
            if predictions[i] == -1:
                truePositives += 1
            else:
                falseNegatives += 1
        else:
            if predictions[i] == -1:
                falsePositives += 1
            else:
                trueNegatives += 1
    return (truePositives, falsePositives, trueNegatives, falseNegatives)

def _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives):
    array = [[truePositives, falsePositives],
            [falseNegatives, trueNegatives]]
    df_cm = pd.DataFrame(
                array,
                index=pd.Index([params.FOLDER_ANOMALOUS, params.FOLDER_NORMAL], name="Predicted"),
                columns= pd.Index([params.FOLDER_ANOMALOUS, params.FOLDER_NORMAL], name="Actual"))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")

def _printAccuracy(truePositives, falsePositives, trueNegatives, falseNegatives):
    correctPredictionAmount = truePositives + trueNegatives
    incorrectPredictionAmount = falsePositives + falseNegatives
    accuracy = correctPredictionAmount / (correctPredictionAmount + incorrectPredictionAmount) * 100
    print("Accuracy: %.2f%%" % accuracy)

def run():
    print("applying PCA to train samples...")
    xTrain = _getTrainSamples()
    pca = _getPca(xTrain)
    print("training OC-SVM...")
    ocSvm = _trainOcSvm(xTrain, pca)
    print("evaluating OC-SVM...")
    xTest, yTest = _getTestSamples()
    predictions = _predict(xTest, pca, ocSvm)
    (truePositives, falsePositives, trueNegatives, falseNegatives) = _evaluate(yTest, predictions)
    print("done!")
    _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives)
    _printAccuracy(truePositives, falsePositives, trueNegatives, falseNegatives)
    plt.show()

if __name__ == "__main__":
    run()
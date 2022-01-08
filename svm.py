import numpy as np
import pandas as pd
import seaborn as sn
import os, sys, random
import base.params as params
import matplotlib.pyplot as plt
import base.constants as consts
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import PCA

def _getFilePaths(datasetPath, subset, isAnomalous):
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
    
    folder = os.path.join(datasetPath,
                        subsetFolder,
                        classFolder)
    return [os.path.join(folder, img) for img in os.listdir(folder)]

def _getPca(numberOfComponents, x):
    pca = PCA(n_components=numberOfComponents)
    pca.fit(x)
    return pca

def _getSamples(datasetPath, subset):
    anomalousFiles = _getFilePaths(datasetPath, subset, True)
    normalFiles = _getFilePaths(datasetPath, subset, False)
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
    random.seed(8)
    random.shuffle(x)
    random.seed(8)
    random.shuffle(y)
    return x, y

def _trainOcSvm(x, pca):
    transformedX = pca.transform(x)
    ocSvm = OneClassSVM(kernel="sigmoid", gamma="auto")
    ocSvm.fit(transformedX)
    return ocSvm

def _trainSvm(x, y, pca):
    transformedX = pca.transform(x)
    svm = SVC(kernel="sigmoid", gamma="auto")
    svm.fit(transformedX, y)
    return svm

def _predict(x, pca, model):
    predictions = []
    nBatches = int(np.ceil(len(x)/params.BATCH_SIZE))
    for i in range(nBatches):
        initial = i*params.BATCH_SIZE
        final = min(initial + params.BATCH_SIZE, len(x))
        xBatch = x[initial:final]
        transformedX = pca.transform(xBatch)
        predictionsBatch = model.predict(transformedX)
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

def _isOneClass():
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
        if algorithm == "ocsvm":
            return True
        elif algorithm == "svm":
            return False
        else:
            sys.exit("Invalid algorithm in arguments. Please chose either \"svm\" or \"ocsvm\"")
    else:
        sys.exit("Algorithm (\"svm\"/\"ocsvm\") missing in arguments")

def _getNumberOfComponents():
    if len(sys.argv) > 2:
        return sys.argv[2]
    else:
        sys.exit("Number of components missing in arguments")

def _getDatasetPath():
    if len(sys.argv) > 3:
        return sys.argv[3]
    else:
        sys.exit("Dataset path missing in arguments")

def run():
    isOneClass = _isOneClass()
    algorithmName = "OC-SVM" if isOneClass else "SVM"
    numberOfComponents = _getNumberOfComponents()
    datasetPath = _getDatasetPath()
    print("Number of components: " + str(numberOfComponents))
    print("applying PCA to train samples...")
    xTrain, yTrain = _getSamples(datasetPath, consts.SetEnum.train)
    pca = _getPca(numberOfComponents, xTrain)
    print("training " + algorithmName + "...")
    if isOneClass:
        model = _trainOcSvm(xTrain, pca)
    else:
        model = _trainSvm(xTrain, yTrain, pca)
    print("evaluating " + algorithmName + "...")
    xTest, yTest = _getSamples(datasetPath, consts.SetEnum.test)
    predictions = _predict(xTest, pca, model)
    (truePositives, falsePositives, trueNegatives, falseNegatives) = _evaluate(yTest, predictions)
    print("done!")
    _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives)
    _printAccuracy(truePositives, falsePositives, trueNegatives, falseNegatives)
    plt.show()

if __name__ == "__main__":
    run()
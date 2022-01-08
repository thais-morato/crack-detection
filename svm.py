import numpy as np
import pandas as pd
import seaborn as sn
import os, sys, random, cv2
import base.params as params
import matplotlib.pyplot as plt
import base.constants as consts
from sklearn.svm import OneClassSVM, SVC
from sklearn.decomposition import IncrementalPCA

def _getFilePaths(datasetPath, subset, isAnomalous):
    if subset == consts.SetEnum.train:
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

def _getSamplesPath(datasetPath, subset):
    normalFiles = _getFilePaths(datasetPath, subset, False)
    anomalousFiles = _getFilePaths(datasetPath, subset, True)
    xPaths = normalFiles + anomalousFiles
    y = [1 for _ in normalFiles] + [-1 for _ in anomalousFiles]
    random.seed(8)
    random.shuffle(xPaths)
    random.seed(8)
    random.shuffle(y)
    return xPaths, y

def _getBatchSize(numberOfComponents):
    batchSize = max(numberOfComponents, params.BATCH_SIZE)
    return batchSize

def _getBatches(xPaths, batchSize):
    nBatches = int(np.ceil(len(xPaths)/batchSize))
    return list(range(nBatches))

def _getBatch(xPaths, y, batchSize, batch):
    initial = batch*batchSize
    final = min(initial + batchSize, len(xPaths))
    batchPaths = xPaths[initial:final]
    xBatch = []
    for file in batchPaths:
        img = cv2.imread(file)
        img = cv2.resize(img, consts.TARGET_IMAGE_SIZE)
        img = np.array(img).flatten()
        xBatch.append(img)
    yBatch = y[initial:final]
    return xBatch, yBatch

def _getPca(xPaths, y, numberOfComponents, batchSize):
    batches = _getBatches(xPaths, batchSize)
    pca = IncrementalPCA(n_components=numberOfComponents, batch_size=batchSize)
    for batch in batches:
        xBatch, yBatch = _getBatch(xPaths, y, batchSize, batch)
        pca.partial_fit(xBatch)
    return pca

def _getSamples(xPaths, y, pca, batchSize):
    batches = _getBatches(xPaths, batchSize)
    x = []
    for batch in batches:
        xBatch, yBatch = _getBatch(xPaths, y, batchSize, batch)
        xTransformedBatch = pca.transform(xBatch)
        x.extend(xTransformedBatch)
    return x, y

def _trainOcSvm(x):
    ocSvm = OneClassSVM(kernel="linear", gamma="scale")
    ocSvm.fit(x)
    return ocSvm

def _trainSvm(x, y):
    svm = SVC(kernel="linear", gamma="scale")
    svm.fit(x, y)
    return svm

def _predict(xTest, model):
    predictions = model.predict(xTest)
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

def _getDatasetPath():
    if len(sys.argv) > 2:
        return sys.argv[2]
    else:
        sys.exit("Dataset path missing in arguments")

def _getNumberOfComponents():
    if len(sys.argv) > 3:
        return int(sys.argv[3])
    else:
        sys.exit("Number of components missing in arguments")

def run():
    isOneClass = _isOneClass()
    algorithmName = "OC-SVM" if isOneClass else "SVM"
    datasetPath = _getDatasetPath()
    numberOfComponents = _getNumberOfComponents()
    batchSize = _getBatchSize(numberOfComponents)
    print("number of components: " + str(numberOfComponents))

    print("training PCA...")
    xTrainPaths, yTrain = _getSamplesPath(datasetPath, consts.SetEnum.train)
    pca = _getPca(xTrainPaths, yTrain, numberOfComponents, batchSize)

    print("training " + algorithmName + "...")
    xTrain, yTrain = _getSamples(xTrainPaths, yTrain, pca, batchSize)
    if isOneClass:
        model = _trainOcSvm(xTrain, yTrain)
    else:
        model = _trainSvm(xTrain, yTrain)
    
    print("evaluating " + algorithmName + "...")
    xTestPaths, yTest = _getSamplesPath(datasetPath, consts.SetEnum.test)
    xTest, yTest = _getSamples(xTestPaths, yTest, pca, batchSize)
    predictions = _predict(xTest, model)
    (truePositives, falsePositives, trueNegatives, falseNegatives) = _evaluate(yTest, predictions)

    print("done!")
    _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives)
    _printAccuracy(truePositives, falsePositives, trueNegatives, falseNegatives)
    plt.show()

if __name__ == "__main__":
    run()
import numpy as np
import pandas as pd
import seaborn as sn
import os, sys, random, cv2
import base.params as params
import matplotlib.pyplot as plt
import base.constants as consts
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDOneClassSVM
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, f1_score

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

def _getBatches(xPaths, batchSize, numberOfComponents, forPca = False):
    if forPca:
        nBatches = int(np.floor(len(xPaths)/batchSize))
        if len(xPaths) - nBatches*batchSize >= numberOfComponents:
            nBatches += 1
    else:
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
    batches = _getBatches(xPaths, batchSize, numberOfComponents, forPca=True)
    pca = IncrementalPCA(n_components=numberOfComponents, batch_size=batchSize)
    for batch in batches:
        xBatch, yBatch = _getBatch(xPaths, y, batchSize, batch)
        pca.partial_fit(xBatch)
    return pca

def _getSamples(xPaths, y, pca, batchSize, numberOfComponents):
    batches = _getBatches(xPaths, batchSize, numberOfComponents)
    x = []
    for batch in batches:
        xBatch, yBatch = _getBatch(xPaths, y, batchSize, batch)
        xTransformedBatch = pca.transform(xBatch)
        x.extend(xTransformedBatch)
    return x, y[:len(x)]

def _getScalingFactor(xTrain):
    norms = [np.linalg.norm(x) for x in xTrain]
    averageNorm = np.average(norms)
    return 1/averageNorm

def _scale(x, scalingFactor):
    scaledX = [[feature*scalingFactor for feature in data] for data in x]
    return scaledX

def _performGridSeach(xTrain, xValidation, yValidation, metric):
    nuValues = [(x+1)/10 for x in range(10)]
    scores = []
    bestModel = None
    for nu in nuValues:
        sgdOcSvm = _trainSgdOcSvm(xTrain, nu)
        predictions = _predict(xValidation, sgdOcSvm)
        if metric == consts.MetricEnum.accuracy:
            score = accuracy_score(yValidation, predictions) * 100
        else:
            score = f1_score(yValidation, predictions)
        if bestModel == None or score > max(scores):
            bestModel = sgdOcSvm
        scores.append(score)
    _plotGridSearchScores(nuValues, scores, metric)
    return bestModel

def _plotGridSearchScores(nuValues, scores, metric):
    plt.figure(figsize=(11, 5))
    plt.bar([str(nu) for nu in nuValues], scores)
    plt.title('Análise do hiperparâmetro \'nu\'')
    plt.xlabel('nu')
    yLabel = 'Acurácia (%)' if metric == consts.MetricEnum.accuracy else 'F1 Score'
    plt.ylabel(yLabel)
    plt.show()

def _trainSgdOcSvm(x, nu):
    sgdOcSvm = SGDOneClassSVM(nu=nu)
    sgdOcSvm.fit(x)
    return sgdOcSvm

def _trainGnBayes(x, y):
    gnBayes = GaussianNB()
    gnBayes.fit(x, y)
    return gnBayes

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

def _printAccuracy(yTest, predictions):
    accuracy = accuracy_score(yTest, predictions) * 100
    print("Accuracy: %.2f%%" % accuracy)

def _getAlgorithm():
    algorithmNames = [algEnum.name for algEnum in consts.AlgorithmEnum]
    if len(sys.argv) > 1:
        algorithmName = sys.argv[1]
        if algorithmName in algorithmNames:
            return consts.AlgorithmEnum[algorithmName]
        else:
            sys.exit("Invalid algorithm in arguments. Please chose one of the following: " + ", ".join(algorithmNames))            
    else:
        sys.exit("Algorithm missing in arguments. Please chose one of the following: " + ", ".join(algorithmNames))

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

def _getGridSearchMetric():
    metricNames = [metric.name for metric in consts.MetricEnum]
    if len(sys.argv) > 4:
        metricName = sys.argv[4]
        if metricName in metricNames:
            return consts.MetricEnum[metricName]
        else:
            sys.exit("Invalid metric in arguments. Please chose one of the following: " + ", ".join(metricNames))            
    else:
        return None

def run():
    algorithm = _getAlgorithm()
    datasetPath = _getDatasetPath()
    numberOfComponents = _getNumberOfComponents()
    batchSize = _getBatchSize(numberOfComponents)
    gridSearchMetric = _getGridSearchMetric()
    print("number of components: " + str(numberOfComponents))

    print("training PCA...")
    xTrainPaths, yTrain = _getSamplesPath(datasetPath, consts.SetEnum.train)
    pca = _getPca(xTrainPaths, yTrain, numberOfComponents, batchSize)

    print("training " + algorithm.name.upper() + "...")
    xTrain, yTrain = _getSamples(xTrainPaths, yTrain, pca, batchSize, numberOfComponents)
    if algorithm == consts.AlgorithmEnum.sgdocsvm:
        scalingFactor = _getScalingFactor(xTrain)
        xTrain = _scale(xTrain, scalingFactor)
        if gridSearchMetric != None:
            xValidationPaths, yValidation = _getSamplesPath(datasetPath, consts.SetEnum.validation)
            xValidation, yValidation = _getSamples(xValidationPaths, yValidation, pca, batchSize, numberOfComponents)
            xValidation = _scale(xValidation, scalingFactor)
            model = _performGridSeach(xTrain, xValidation, yValidation, gridSearchMetric)
        else:
            model = _trainSgdOcSvm(xTrain, 0.5)
    else: # consts.AlgorithmEnum.gnbayes
        scalingFactor = None
        model = _trainGnBayes(xTrain, yTrain)
    
    print("evaluating " + algorithm.name.upper() + "...")
    xTestPaths, yTest = _getSamplesPath(datasetPath, consts.SetEnum.test)
    xTest, yTest = _getSamples(xTestPaths, yTest, pca, batchSize, numberOfComponents)
    if scalingFactor != None:
        xTest = _scale(xTest, scalingFactor)
    predictions = _predict(xTest, model)
    (truePositives, falsePositives, trueNegatives, falseNegatives) = _evaluate(yTest, predictions)

    print("done!")
    _plotConfusionMatrix(truePositives, falsePositives, trueNegatives, falseNegatives)
    _printAccuracy(yTest, predictions)
    plt.show()

if __name__ == "__main__":
    run()
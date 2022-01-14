import numpy as np
import os, sys, random, cv2
import matplotlib.pyplot as plt
import base.constants as consts
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

def _getNumberOfComponents(xTrainPaths):
    exampleImg = _readImg(xTrainPaths[0])
    return min(len(xTrainPaths), exampleImg.shape[0])

def _getBatches(xPaths, batchSize, numberOfComponents):
    nBatches = int(np.floor(len(xPaths)/batchSize))
    if len(xPaths) - nBatches*batchSize >= numberOfComponents:
        nBatches += 1
    return list(range(nBatches))

def _getBatch(xPaths, y, batchSize, batch):
    initial = batch*batchSize
    final = min(initial + batchSize, len(xPaths))
    batchPaths = xPaths[initial:final]
    xBatch = []
    for file in batchPaths:
        img = _readImg(file)
        xBatch.append(img)
    yBatch = y[initial:final]
    return xBatch, yBatch

def _readImg(file):
    img = cv2.imread(file)
    img = cv2.resize(img, consts.TARGET_IMAGE_SIZE)
    img = np.array(img).flatten()
    return img

def _getPca(xPaths, y, numberOfComponents, batchSize):
    batches = _getBatches(xPaths, batchSize, numberOfComponents)
    pca = IncrementalPCA(n_components=numberOfComponents, batch_size=batchSize)
    for batch in batches:
        xBatch, yBatch = _getBatch(xPaths, y, batchSize, batch)
        pca.partial_fit(xBatch)
    return pca

def _getExplainedVariance(pca):
    nComponents = [x+1 for x in range(len(pca.explained_variance_ratio_))]
    explainedVariance = [y*100 for y in np.cumsum(pca.explained_variance_ratio_)]
    return nComponents, explainedVariance

def _plotExplainedVariance(nComponents, explainedVariance):
    plt.plot(nComponents, explainedVariance)
    plt.title('Análise das componentes PCA')
    plt.xlabel('Número de componentes PCA')
    plt.ylabel('Variância explicada (%% acumulada)')
    plt.show()

def _printNumberOfComponents(nComponents, explainedVariance):
    goals = [50, 70, 80, 90, 95, 99]
    npExplainedVariance = np.asarray(explainedVariance)
    print('explained variance:')
    for goal in goals:
        indexes = np.where(npExplainedVariance >= goal)[0]
        if len(indexes) == 0:
            print(str(goal) + '%: not reached')
        else:
            i = indexes[0]
            print(str(goal) + '%: ' + str(nComponents[i]))

def _getDatasetPath():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        sys.exit("Dataset path missing in arguments")

def run():
    datasetPath = _getDatasetPath()

    print("analysing PCA...")
    xTrainPaths, yTrain = _getSamplesPath(datasetPath, consts.SetEnum.train)
    numberOfComponents = _getNumberOfComponents(xTrainPaths)
    batchSize = numberOfComponents
    pca = _getPca(xTrainPaths, yTrain, numberOfComponents, batchSize)
    nComponents, explainedVariance = _getExplainedVariance(pca)
    _plotExplainedVariance(nComponents, explainedVariance)
    _printNumberOfComponents(nComponents, explainedVariance)

if __name__ == "__main__":
    run()
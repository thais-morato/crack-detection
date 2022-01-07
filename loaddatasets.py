"""

    Run this file to load the train, validation and test datasets with the samples
    from your database.
    
    For this script to work properly, you should change the ./base/params.py file
    with the parameters you wish to consider. This parameters include the path to
    where your database is stored, the percentage of samples for each set (which
    should sum up to 1) and the size of the images in the dataset.

    The given database should have the following structure:

    params.PATH_DATABASE
    |___params.FOLDER_ANOMALOUS
    |___params.FOLDER_NORMAL

    The train, validation and test sets will be created with the following structure:

    ./dataset
    |___test
    |   |___crack
    |   |___no-crack
    |___train
    |   |___crack
    |   |___no-crack
    |___validation
        |___crack
        |___no-crack

"""

import sys
import random, os, shutil
import base.params as params
import base.constants as consts

def _loadDatasets(pathDataset, samplesPath, classFolder):
    files = os.listdir(samplesPath)
    amount = len(files)
    random.shuffle(files)

    trainLimit = amount * params.PERCENTAGE_TRAIN
    validationLimit = amount * params.PERCENTAGE_VALIDATION + trainLimit
    for i, fileName in enumerate(files):
        sourcePath = os.path.join(samplesPath, fileName)

        if i + 1 <= trainLimit:
            destinationPath = os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_TRAIN,
                                classFolder)
        elif i + 1 <= validationLimit:
            destinationPath = os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_VALIDATION,
                                classFolder)
        else:
            destinationPath = os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_TEST,
                                classFolder)
        
        shutil.copy(sourcePath, destinationPath)

def _createFolderStructure(pathDataset):
    if os.path.isdir(pathDataset):
        shutil.rmtree(pathDataset)

    for path in [
        pathDataset,
        os.path.join(pathDataset, consts.AP_FOLDER_TRAIN),
        os.path.join(pathDataset, consts.AP_FOLDER_TRAIN, consts.AP_FOLDER_NORMAL),
        os.path.join(pathDataset, consts.AP_FOLDER_TRAIN, consts.AP_FOLDER_ANOMALOUS),
        os.path.join(pathDataset, consts.AP_FOLDER_VALIDATION),
        os.path.join(pathDataset, consts.AP_FOLDER_VALIDATION, consts.AP_FOLDER_NORMAL),
        os.path.join(pathDataset, consts.AP_FOLDER_VALIDATION, consts.AP_FOLDER_ANOMALOUS),
        os.path.join(pathDataset, consts.AP_FOLDER_TEST),
        os.path.join(pathDataset, consts.AP_FOLDER_TEST, consts.AP_FOLDER_NORMAL),
        os.path.join(pathDataset, consts.AP_FOLDER_TEST, consts.AP_FOLDER_ANOMALOUS)]:
        if not os.path.isdir(path):
            os.mkdir(path)

def _printStatistics(pathDataset):
    statistics = """
        {PATH_DATASET}
        |___{FOLDER_TEST}
        |   |___{FOLDER_ANOMALOUS} ({testAnomalousAmount} samples)
        |   |___{FOLDER_NORMAL} ({testNormalAmount} samples)
        |___{FOLDER_TRAIN}
        |   |___{FOLDER_ANOMALOUS} ({trainAnomalousAmount} samples)
        |   |___{FOLDER_NORMAL} ({trainNormalAmount} samples)
        |___{FOLDER_VALIDATION}
            |___{FOLDER_ANOMALOUS} ({validationAnomalousAmount} samples)
            |___{FOLDER_NORMAL} ({validationNormalAmount} samples)
    """.format(
        PATH_DATASET = pathDataset,
        FOLDER_TEST = consts.AP_FOLDER_TEST,
        FOLDER_TRAIN = consts.AP_FOLDER_TRAIN,
        FOLDER_VALIDATION = consts.AP_FOLDER_VALIDATION,
        FOLDER_NORMAL = consts.AP_FOLDER_NORMAL,
        FOLDER_ANOMALOUS = consts.AP_FOLDER_ANOMALOUS,
        testAnomalousAmount = len(os.listdir(os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_TEST,
                                consts.AP_FOLDER_ANOMALOUS))),
        testNormalAmount = len(os.listdir(os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_TEST,
                                consts.AP_FOLDER_NORMAL))),
        trainAnomalousAmount = len(os.listdir(os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_TRAIN,
                                consts.AP_FOLDER_ANOMALOUS))),
        trainNormalAmount = len(os.listdir(os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_TRAIN,
                                consts.AP_FOLDER_NORMAL))),
        validationAnomalousAmount = len(os.listdir(os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_VALIDATION,
                                consts.AP_FOLDER_ANOMALOUS))),
        validationNormalAmount = len(os.listdir(os.path.join(
                                pathDataset,
                                consts.AP_FOLDER_VALIDATION,
                                consts.AP_FOLDER_NORMAL))))

    print(statistics)

def _undersampleAnomaliesInTrainingSet(pathDataset, undersamplingPercentage):
    normalFolder = os.path.join(pathDataset, consts.AP_FOLDER_TRAIN, consts.AP_FOLDER_NORMAL)
    anomalousFolder = os.path.join(pathDataset, consts.AP_FOLDER_TRAIN, consts.AP_FOLDER_ANOMALOUS)

    normalSamples = os.listdir(normalFolder)
    anomalousSamples = os.listdir(anomalousFolder)

    # calculating how many anomalous samples to delete (d) to reach the desired percentage
    n = len(normalSamples)
    a = len(anomalousSamples)
    r = undersamplingPercentage/100
    d = int(max(0, r*n/(r-1)+a))

    random.shuffle(anomalousSamples)
    anomalousSamplesToDelete = anomalousSamples[:d]

    for anomalousSampleToDelete in anomalousSamplesToDelete:
        os.remove(os.path.join(anomalousFolder, anomalousSampleToDelete))

def _getUndersamplingAnomaliesPercentages():
    if len(sys.argv) > 1:
        return [float(x) for x in sys.argv[1:]]
    else:
        return None

def percentageToString(x):
    if float.is_integer(x):
        return '{:0>2}'.format(int(x))
    else:
        return '{:0>5.2f}'.format(x)

def _loadDatasetsWithoutUndersampling(normalSamplesPath, anomalousSamplesPath):
    print("dataset: " + consts.AP_PATH_DATASET)
    print("creating folder structure...")
    _createFolderStructure(consts.AP_PATH_DATASET)
    print("populating with normal samples...")
    _loadDatasets(consts.AP_PATH_DATASET, normalSamplesPath, consts.AP_FOLDER_NORMAL)
    print("populating with anomalous samples...")
    _loadDatasets(consts.AP_PATH_DATASET, anomalousSamplesPath, consts.AP_FOLDER_ANOMALOUS)
    print("done!")

    _printStatistics(consts.AP_PATH_DATASET)

def _loadDatasetsUndersamplingAnomalies(normalSamplesPath, anomalousSamplesPath, undersamplingPercentage):
    pathDataset = consts.AP_PATH_DATASET + "-an-" + percentageToString(undersamplingPercentage)
    print("dataset: " + pathDataset)
    print("creating folder structure...")
    _createFolderStructure(pathDataset)
    print("populating with normal samples...")
    _loadDatasets(pathDataset, normalSamplesPath, consts.AP_FOLDER_NORMAL)
    print("populating with anomalous samples...")
    _loadDatasets(pathDataset, anomalousSamplesPath, consts.AP_FOLDER_ANOMALOUS)
    print("undersampling anomalies in training set...")
    _undersampleAnomaliesInTrainingSet(pathDataset, undersamplingPercentage)
    print("done!")

    _printStatistics(pathDataset)

def run():
    random.seed(params.LOAD_DATASETS_SEED)

    undersamplingAnomaliesPercentages = _getUndersamplingAnomaliesPercentages()

    normalSamplesPath = os.path.join(params.PATH_DATABASE, params.FOLDER_NORMAL)
    anomalousSamplesPath = os.path.join(params.PATH_DATABASE, params.FOLDER_ANOMALOUS)

    if(undersamplingAnomaliesPercentages == None):
        _loadDatasetsWithoutUndersampling(normalSamplesPath, anomalousSamplesPath)
    else:
        for undersamplingPercentage in undersamplingAnomaliesPercentages:
            _loadDatasetsUndersamplingAnomalies(normalSamplesPath, anomalousSamplesPath, undersamplingPercentage)

if __name__ == "__main__":
    if params.PERCENTAGE_TRAIN + params.PERCENTAGE_VALIDATION + params.PERCENTAGE_TEST != 1:
        sys.exit("PERCENTAGE_TRAIN, PERCENTAGE_VALIDATION and PERCENTAGE_TEST in params.py " +
            "should sum up to 1")
    run()
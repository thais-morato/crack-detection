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

import random, os, shutil
import base.params as params
import base.constants as consts

def _loadDatasets(samplesPath, classFolder):
    files = os.listdir(samplesPath)
    amount = len(files)
    random.shuffle(files)

    trainLimit = amount * params.PERCENTAGE_TRAIN
    validationLimit = amount * params.PERCENTAGE_VALIDATION + trainLimit
    for i, fileName in enumerate(files):
        sourcePath = os.path.join(samplesPath, fileName)

        if i + 1 <= trainLimit:
            destinationPath = os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_TRAIN,
                                classFolder)
        elif i + 1 <= validationLimit:
            destinationPath = os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_VALIDATION,
                                classFolder)
        else:
            destinationPath = os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_TEST,
                                classFolder)
        
        shutil.copy(sourcePath, destinationPath)

def _createFolderStructure():
    if os.path.isdir(consts.AP_PATH_DATASET):
        shutil.rmtree(consts.AP_PATH_DATASET)

    for path in [
        consts.AP_PATH_DATASET,
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_TRAIN),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_TRAIN, consts.AP_FOLDER_NORMAL),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_TRAIN, consts.AP_FOLDER_ANOMALOUS),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_VALIDATION),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_VALIDATION, consts.AP_FOLDER_NORMAL),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_VALIDATION, consts.AP_FOLDER_ANOMALOUS),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_TEST),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_TEST, consts.AP_FOLDER_NORMAL),
        os.path.join(consts.AP_PATH_DATASET, consts.AP_FOLDER_TEST, consts.AP_FOLDER_ANOMALOUS)]:
        if not os.path.isdir(path):
            os.mkdir(path)

def _printStatistics():
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
        PATH_DATASET = consts.AP_PATH_DATASET,
        FOLDER_TEST = consts.AP_FOLDER_TEST,
        FOLDER_TRAIN = consts.AP_FOLDER_TRAIN,
        FOLDER_VALIDATION = consts.AP_FOLDER_VALIDATION,
        FOLDER_NORMAL = consts.AP_FOLDER_NORMAL,
        FOLDER_ANOMALOUS = consts.AP_FOLDER_ANOMALOUS,
        testAnomalousAmount = len(os.listdir(os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_TEST,
                                consts.AP_FOLDER_ANOMALOUS))),
        testNormalAmount = len(os.listdir(os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_TEST,
                                consts.AP_FOLDER_NORMAL))),
        trainAnomalousAmount = len(os.listdir(os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_TRAIN,
                                consts.AP_FOLDER_ANOMALOUS))),
        trainNormalAmount = len(os.listdir(os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_TRAIN,
                                consts.AP_FOLDER_NORMAL))),
        validationAnomalousAmount = len(os.listdir(os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_VALIDATION,
                                consts.AP_FOLDER_ANOMALOUS))),
        validationNormalAmount = len(os.listdir(os.path.join(
                                consts.AP_PATH_DATASET,
                                consts.AP_FOLDER_VALIDATION,
                                consts.AP_FOLDER_NORMAL))))

    print(statistics)

def run():
    random.seed(11)

    normalSamplesPath = os.path.join(params.PATH_DATABASE, params.FOLDER_NORMAL)
    anomalousSamplesPath = os.path.join(params.PATH_DATABASE, params.FOLDER_ANOMALOUS)

    print("creating folder structure...")
    _createFolderStructure()
    print("populating with normal samples...")
    _loadDatasets(normalSamplesPath, consts.AP_FOLDER_NORMAL)
    print("populating with anomalous samples...")
    _loadDatasets(anomalousSamplesPath, consts.AP_FOLDER_ANOMALOUS)
    print("done!")

    _printStatistics()

if __name__ == "__main__":
    run()
"""

    Run this file to generate the training and testing datasets.
    
    For this script to work properly, you should change the ./base/params.py file
    with the parameters you wish to consider. This parameters include the path to
    where your database is stored, the percentage of training samples and so on. 

"""

import random, os, shutil
from base.constants import *
from base.params import *

def generateDataset(samplesPath, classFolder):
    files = os.listdir(samplesPath)
    amount = len(files)
    random.shuffle(files)

    trainLimit = amount * PERCENTAGE_TRAINABLE
    for i, fileName in enumerate(files):
        sourcePath = os.path.join(samplesPath, fileName)
        destinationPath = AP_PATH_DATASET

        if i + 1 <= trainLimit:
            destinationPath = os.path.join(destinationPath, AP_FOLDER_TRAIN, classFolder)
        else:
            destinationPath = os.path.join(destinationPath, AP_FOLDER_TEST, classFolder)
        
        shutil.copy(sourcePath, destinationPath)

def createDestinationPath():
    if os.path.isdir(AP_PATH_DATASET):
        shutil.rmtree(AP_PATH_DATASET)

    for path in [
        AP_PATH_DATASET,
        os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN),
        os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN, AP_FOLDER_NORMAL),
        os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN, AP_FOLDER_ANOMALOUS),
        os.path.join(AP_PATH_DATASET, AP_FOLDER_TEST),
        os.path.join(AP_PATH_DATASET, AP_FOLDER_TEST, AP_FOLDER_NORMAL),
        os.path.join(AP_PATH_DATASET, AP_FOLDER_TEST, AP_FOLDER_ANOMALOUS),
    ]:
        if not os.path.isdir(path):
            os.mkdir(path)

def printStatistics():
    statistics = """
        {AP_PATH_DATASET}
            {AP_FOLDER_TRAIN}
                {AP_FOLDER_NORMAL} ({trainNormalAmount} samples)
                {AP_FOLDER_ANOMALOUS} ({trainAnomalousAmount} samples)
            {AP_FOLDER_TEST}
                {AP_FOLDER_NORMAL} ({testNormalAmount} samples)
                {AP_FOLDER_ANOMALOUS} ({testAnomalousAmount} samples)
    """.format(
        AP_PATH_DATASET = AP_PATH_DATASET,
        AP_FOLDER_TRAIN = AP_FOLDER_TRAIN,
        AP_FOLDER_TEST = AP_FOLDER_TEST,
        AP_FOLDER_NORMAL = AP_FOLDER_NORMAL,
        AP_FOLDER_ANOMALOUS = AP_FOLDER_ANOMALOUS,
        trainNormalAmount = len(os.listdir(os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN, AP_FOLDER_NORMAL))),
        trainAnomalousAmount = len(os.listdir(os.path.join(AP_PATH_DATASET, AP_FOLDER_TRAIN, AP_FOLDER_ANOMALOUS))),
        testNormalAmount = len(os.listdir(os.path.join(AP_PATH_DATASET, AP_FOLDER_TEST, AP_FOLDER_NORMAL))),
        testAnomalousAmount = len(os.listdir(os.path.join(AP_PATH_DATASET, AP_FOLDER_TEST, AP_FOLDER_ANOMALOUS)))
    )

    print(statistics)

def run():
    random.seed(11)

    normalSamplesPath = os.path.join(PATH_DATASET, FOLDER_NORMAL)
    anomalousSamplesPath = os.path.join(PATH_DATASET, FOLDER_ANOMALOUS)

    print("creating clean datasets...")
    createDestinationPath()
    print("populating with normal samples...")
    generateDataset(normalSamplesPath, AP_FOLDER_NORMAL)
    print("populating with anomalous samples...")
    generateDataset(anomalousSamplesPath, AP_FOLDER_ANOMALOUS)
    print("done!")

    printStatistics()

if __name__ == "__main__":
    run()
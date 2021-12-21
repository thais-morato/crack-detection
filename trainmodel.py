"""

    Run this file to train a CNN Model based on the datasets you created previously
    with loaddatasets.py.
    
"""

import sys
import base.params as params
import base.constants as consts
import training.training as train
import preprocessing.preprocessing as prep

def _isTrainingFromScratch():
    while True:
        option = input("Would you like to train model from scratch or resume training? " +
            "[" + consts.TRAIN_OPTION_SCRATCH + "/" + consts.TRAIN_OPTION_RESUME + "/" +
            consts.OPTION_CANCEL + "]: ")
        if option == consts.OPTION_CANCEL:
            sys.exit()
        if option == consts.TRAIN_OPTION_SCRATCH or option == consts.TRAIN_OPTION_RESUME:
            break
    return option == consts.TRAIN_OPTION_SCRATCH

def run():
    isTrainingFromScratch = _isTrainingFromScratch()

    print("fetching train and validation datasets...")
    if(params.APPLY_PREPROCESSING):
        trainSet = prep.getPreprocessedDataset(set=consts.SetEnum.train, applyDataAugmentation=params.APPLY_DATA_AUGMENTATION)
        validationSet = prep.getPreprocessedDataset(set=consts.SetEnum.validation, applyDataAugmentation=False)
    else:
        trainSet = prep.getRawDataset(set=consts.SetEnum.train, applyDataAugmentation=params.APPLY_DATA_AUGMENTATION)
        validationSet = prep.getRawDataset(set=consts.SetEnum.validation, applyDataAugmentation=False)

    print("starting training...")
    if isTrainingFromScratch:
        if params.IS_FINE_TUNING:
            train.fineTuneModelFromScratch(trainSet, validationSet)
        else:
            train.trainModelFromScratch(trainSet, validationSet)
    else:
        if params.IS_FINE_TUNING:
            train.resumeFineTuning(trainSet, validationSet)
        else:
            train.resumeTraining(trainSet, validationSet)
        
    print("done!")
    
if __name__ == "__main__":
    run()


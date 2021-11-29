"""

    TODO: Documentation

"""

import sys, os
sys.path.insert(1, os.curdir)

import base.params as params
import base.constants as consts
import matplotlib.pyplot as plt
import preprocessing.preprocessing as prep

class _Plot:
    __figures = 0

    def plotExamples(self, set):
        plot = 0
        fig = plt.figure(self.__figures, figsize=(7.8, 5))
        self.__figures += 1
        images, labels = next(iter(set))
        for images, labels in set:
            for i in range(params.BATCH_SIZE):
                plt.subplot(3, 5, plot + 1)
                plt.imshow(images[i].astype('uint8'))
                plt.title(_getClassName(set.class_indices, labels[i]))
                plt.axis("off")
                plot += 1
                if plot >= 15:
                    break
            if plot >= 15:
                    break
        fig.show()
    
    def showPlots(self):
        plt.show()

def _getClassName(class_indices, value):
    return list(class_indices.keys())[list(class_indices.values()).index(value)]

def run():
    plot = _Plot()
    trainSet = prep.getPreprocessedDataset(set=consts.SetEnum.train, applyDataAugmentation=False)
    plot.plotExamples(trainSet)
    rawTrainSet = prep.getRawDataset(set=consts.SetEnum.train, applyDataAugmentation=False)
    plot.plotExamples(rawTrainSet)
    plot.showPlots()

if __name__ == "__main__":
    run()

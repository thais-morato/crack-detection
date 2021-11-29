"""

    TODO: Documentation

"""

import sys, os
sys.path.insert(1, os.curdir)

import random
import base.params as params
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class _Plot:
    __figures = 0

    def plotExamples(self, classFolder, fileList, title):
        fig = plt.figure(self.__figures, figsize=(7, 3.5))
        self.__figures += 1
        for i in range(8):
            ax = fig.add_subplot(2,4,i+1)
            ax.set_title("Amostra " + str(i+1), size=10)
            img = mpimg.imread(os.path.join(
                params.PATH_DATABASE,
                classFolder,
                fileList[i]))
            ax.imshow(img)
            plt.axis('off')
        plt.subplots_adjust(bottom=0, top=0.8)
        plt.suptitle(title)
        fig.show()
    
    def showPlots(self):
        plt.show()

def _getFileList(classFolder):
    path = os.path.join(params.PATH_DATABASE, classFolder)
    fileList = os.listdir(path)
    random.shuffle(fileList)
    return fileList

def run():
    random.seed(10)
    normalFileList = _getFileList(params.FOLDER_NORMAL)
    anomalousFileList = _getFileList(params.FOLDER_ANOMALOUS)
    plot = _Plot()
    plot.plotExamples(params.FOLDER_NORMAL, normalFileList, "Amostras normais")
    plot.plotExamples(params.FOLDER_ANOMALOUS,anomalousFileList, "Amostras anômalas")
    plot.showPlots()

if __name__ == "__main__":
    run()
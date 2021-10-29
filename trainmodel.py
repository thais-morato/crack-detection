"""

    Run this file to train a CNN Model based on the datasets you created previously
    with loaddatasets.py.
    
"""

from preprocessing.preprocessing import preprocess
from training.training import train
import matplotlib.pyplot as plt

def run():
    print("preprocessing inputs...")
    (trainSet, testSet) = preprocess()
    print("training convolutional neural network...")
    (modelHistory, trainingDuration) = train(trainSet, testSet)
    print("done!")

    printStatistics(modelHistory, trainingDuration)

def printStatistics(modelHistory, trainingDuration):
    statistics = """
        training completed in: {trainingDuration}
    """.format(trainingDuration = trainingDuration)
    print(statistics)

    plt.plot(modelHistory.history['accuracy'])
    plt.plot(modelHistory.history['val_accuracy'])
    plt.title('CNN Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    run()


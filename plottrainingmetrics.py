"""

    Run this file to visualize the training metrics of the model.
    
"""

import os, csv
import base.params as params
import base.constants as consts
import matplotlib.pyplot as plt

def _getLogPath():
    return os.path.join(consts.AP_FOLDER_LOG, params.MODEL_NAME + consts.EXTENSION_LOG)

def _readLog(path):
    epoch = []
    accuracy = []
    loss = []
    val_accuracy = []
    val_loss = []

    metrics = (epoch, accuracy, loss, val_accuracy, val_loss)

    file = open(path)
    reader = csv.reader(file)
    isHeader = True
    for row in reader:
        if isHeader:
            isHeader = False
            continue
        
        for i, metric in enumerate(metrics):
            metric.append(row[i])
    
    return metrics

def run():
    path = _getLogPath()
    metrics = _readLog(path)
    _plotMetrics(metrics)

def _plotMetrics(metrics):
    (epoch, accuracy, loss, val_accuracy, val_loss) = metrics
    _plotAccuracyMetrics(epoch, accuracy, val_accuracy)
    _plotLossMetrics(epoch, loss, val_loss)
    plt.show()

def _plotAccuracyMetrics(epoch, accuracy, val_accuracy):
    plt_epoch = [int(x)+1 for x in epoch]
    plt_accuracy = [float(y)*100 for y in accuracy]
    plt_val_accuracy = [float(y)*100 for y in val_accuracy]

    fig = plt.figure('accuracy')
    plt.plot(plt_epoch, plt_accuracy)
    plt.plot(plt_epoch, plt_val_accuracy)
    plt.title('CNN Model accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.xlim(min(plt_epoch), max(plt_epoch))
    plt.ylim(0, 100)
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig.show()

def _plotLossMetrics(epoch, loss, val_loss):
    plt_epoch = [int(x)+1 for x in epoch]
    plt_loss = [float(y) for y in loss]
    plt_val_loss = [float(y) for y in val_loss]

    fig = plt.figure('loss')
    plt.plot(plt_epoch, plt_loss)
    plt.plot(plt_epoch, plt_val_loss)
    plt.title('CNN Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(min(plt_epoch), max(plt_epoch))
    plt.legend(['Training', 'Validation'], loc='upper left')
    fig.show()

if __name__ == "__main__":
    run()


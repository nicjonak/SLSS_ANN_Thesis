#Data Visualizer
#Functions to plot training/Validation curves

import numpy as np
import matplotlib.pyplot as plt


def plot_training(num_sbplt, cur_sbplt, outc, save_num):
    path = "../"+outc+"Net"+str(save_num)

    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))

    ne = len(train_acc)

    x_pnts = np.arange(ne)

    plt.subplot(num_sbplt, 2, cur_sbplt)
    plt.title("Train vs Validation Accuracy")
    plt.plot(x_pnts, train_acc, label="Train")
    plt.plot(x_pnts, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')

    plt.subplot(num_sbplt, 2, cur_sbplt + 1)
    plt.title("Train vs Validation Loss")
    plt.plot(x_pnts, train_loss, label="Train")
    plt.plot(x_pnts, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')

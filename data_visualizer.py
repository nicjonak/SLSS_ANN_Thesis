#Data Visualizer
#Functions to plot training/Validation curves

import numpy as np
import matplotlib.pyplot as plt

#Plot Training and Validation curves (Specified by outc and save_num)
def plot_training(plt_avg, outc, save_num, one_epoch):
    path = "../"+outc+"Net"+str(save_num)
    #print("path = ", path)

    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))

    if plt_avg:
        ep_train_acc = np.loadtxt("{}_ep_train_acc.csv".format(path))
        ep_val_acc = np.loadtxt("{}_ep_val_acc.csv".format(path))
        ep_train_loss = np.loadtxt("{}_ep_train_loss.csv".format(path))
        ep_val_loss = np.loadtxt("{}_ep_val_loss.csv".format(path))

    ftrain_acc = train_acc.flatten()
    fval_acc = val_acc.flatten()
    ftrain_loss = train_loss.flatten()
    fval_loss = val_loss.flatten()

    if plt_avg:
        init_avg_train_acc = (ftrain_acc[0] + ftrain_acc[1]) / 2
        init_avg_val_acc = (fval_acc[0] + fval_acc[1]) / 2
        init_avg_train_loss = (ftrain_loss[0] + ftrain_loss[1]) / 2
        init_avg_val_loss = (fval_loss[0] + fval_loss[1]) / 2

        ep_train_acc = np.append([init_avg_train_acc], ep_train_acc)
        ep_val_acc = np.append([init_avg_val_acc], ep_val_acc)
        ep_train_loss = np.append([init_avg_train_loss], ep_train_loss)
        ep_val_loss = np.append([init_avg_val_loss], ep_val_loss)

    if one_epoch:
        ne = 1
        #print("ne = ",ne)
        nf = len(train_acc)
        #print("nf = ",nf)
    else:
        ne = len(train_acc)
        #print("ne = ",ne)
        nf = len(train_acc[0])
        #print("nf = ",nf)

    
    x_pnts = np.arange(ne*nf)
    #print("x_pnts = ", x_pnts)
    if plt_avg:
        ep_x_pnts = np.linspace(nf-1, ne*nf-1, ne)
        ep_x_pnts = np.append([0], ep_x_pnts)
    #print("ep_x_pnts = ", ep_x_pnts)

    plt.subplot(1, 2, 1)
    plt.title("Train vs Validation Accuracy")
    plt.plot(x_pnts, ftrain_acc, label="Train")
    plt.plot(x_pnts, fval_acc, label="Validation")
    if plt_avg:
        plt.plot(ep_x_pnts, ep_train_acc, label="Train AVG")
        plt.plot(ep_x_pnts, ep_val_acc, label="Validation AVG")
    plt.xlabel("Epoch/Fold")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    #plt.show()
    plt.subplot(1, 2, 2)
    plt.title("Train vs Validation Loss")
    plt.plot(x_pnts, ftrain_loss, label="Train")
    plt.plot(x_pnts, fval_loss, label="Validation")
    if plt_avg:
        plt.plot(ep_x_pnts, ep_train_loss, label="Train AVG")
        plt.plot(ep_x_pnts, ep_val_loss, label="Validation AVG")
    plt.xlabel("Epoch/Fold")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    #plt.show(block=False)

#plot_training(True, "Recovery", 0)
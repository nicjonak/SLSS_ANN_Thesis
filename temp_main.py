#Temp Main
#Temporary Main file, contains train and test functions that intiate, visualize, and time training and testing

import time
import torch
import random
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from test_nets import *
from net_trainer import *
from data_visualizer import *

#Update and Fix with nets
index_to_net = {
        0: backpain_net(),
        1: backpain_net(),
        2: odiscore_net(),
        3: recovery_net(),
        4: eqidxtl12_net(),
        5: recovery_net()
        }

"""
#OLD CODE PRESERVED
#Trains given net with given hyperparameters
def train(net, batch, ts_per, lrn_rate, mntum, folds, epochs, outc, save_num):
    print("--- Starting Training ---")
    start_time = time.time()
    tv_data, ts_data = load_data(ts_per)
    ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True)

    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    trainNet(tv_data, net, batch, lrn_rate, mntum, folds, epochs, outc, save_num)
    end_time = time.time()
    print("--- Finished Training ---")
    elapsed_time = end_time - start_time
    print("Total Training Time: {:.2f} seconds".format(elapsed_time))
    print()
    one_epoch = True if epochs == 1 else False
    #print("one_epoch = ", one_epoch)
    plot_training(True, outc, save_num, one_epoch)
    return ts_load
"""
"""
#OLD CODE PRESERVED
#Tests net (Specified by outc and save_num) on given dataset [ne, nf used for graphing test lines on the train/val graphs]
def test(net, dataset, outc, save_num, nf, ne):
    print("--- Starting Testing ---")
    path = "../"+outc+"Net"+str(save_num)
    #print("path = ", path)

    outc_idx = outcome_to_index[outc]
    #print("outc = ", outc)
    #print("outc_idx = ", outc_idx)
    if (outc_idx == 3) or (outc_idx == 5):
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    
    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    net.load_state_dict(torch.load(path))
    ts_acc,ts_loss = validate(net, dataset, criterion, outc)
    print(("Test Acc: {}, Test Loss: {}").format(ts_acc, ts_loss))
    print("--- Finished Testing ---")
    x_pnts = np.arange(ne*nf)
    ts_acc_pnts = np.repeat(ts_acc, ne*nf)
    ts_loss_pnts = np.repeat(ts_loss, ne*nf)
    plt.subplot(1, 2, 1)
    plt.plot(x_pnts, ts_acc_pnts, label="Test")
    plt.subplot(1, 2, 2)
    plt.plot(x_pnts, ts_loss_pnts, label="Test")
"""

#WIP function to search range of hyperparameters and report best combination might wake up and completely change/scrap
def hp_search(net, folds, val_per_min, val_per_max, val_per_itr, batch_min, batch_max, batch_itr, lrn_rate_min, lrn_rate_max, lrn_rate_itr, mntum_min, mntum_max, mntum_itr, epochs_min, epochs_max, epochs_itr, outc, save_num):
    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
    
    path = "../"+outc+"Net"+str(save_num)
    print("path = ", path)
    outc_idx = outcome_to_index[outc]
    print("outc = ", outc)
    print("outc_idx = ", outc_idx)

    #print("--- Starting Training ---")
    #start_time = time.time()

    tvs_data = load_data()
    splits = [1/folds] * folds
    tvs_folds = torch.utils.data.random_split(tvs_data, splits)

    print()


    if (cross_val == False):
        print("--- Starting Training ---")
        start_time = time.time()
        fold = random.randrange(folds)

        print("Test Fold: ", fold)

        ts_data = tvs_folds[fold]
        tv_list = tvs_folds.copy()
        tv_list.pop(fold)
        tv_data = torch.utils.data.ConcatDataset(tv_list)

        ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True, drop_last=True)

        tvsplits = len(tv_data)
        val_split = int(tvsplits * val_per)
        trn_split = tvsplits - val_split
        trn_data, val_data = torch.utils.data.random_split(tv_data, [trn_split, val_split])

        val_load = DataLoader(val_data, batch_size=batch, shuffle=True, drop_last=True)
        trn_load = DataLoader(trn_data, batch_size=batch, shuffle=True, drop_last=True)

        #print("len tst_data = ", len(ts_data))
        #print("len trn_data = ", len(trn_data))
        #print("len val_data = ", len(val_data))
        #print()
        #print("Test Fold: ", fold)

        trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num)

        end_time = time.time()
        print("--- Finished Training ---")
        elapsed_time = end_time - start_time
        print("Total Training Time: {:.2f} seconds".format(elapsed_time))
        print()

        plot_training(1, 1, outc, save_num)

        fold_tst_acc, fold_tst_loss = test(net, ts_load, outc, save_num, epochs, 1, 1)

        #average_fold()


        return ts_load

    else:

        print("--- Starting Cross Validation ---")
        start_cv_time = time.time()

        fold_tst_loss = np.zeros(folds)
        fold_tst_acc = np.zeros(folds)
        
        for fold in range(folds):
            print("--- Starting Training ---")
            start_time = time.time()
            print("Fold: ", fold)
            ts_data = tvs_folds[fold]
            tv_list = tvs_folds.copy()
            tv_list.pop(fold)
            tv_data = torch.utils.data.ConcatDataset(tv_list)

            #print("len ts_data = ", len(ts_data))
            #print("len tv_data = ", len(tv_data))

            ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True, drop_last=True)

            tvsplits = len(tv_data)
            val_split = int(tvsplits * val_per)
            trn_split = tvsplits - val_split
            trn_data, val_data = torch.utils.data.random_split(tv_data, [trn_split, val_split])

            val_load = DataLoader(val_data, batch_size=batch, shuffle=True, drop_last=True)
            trn_load = DataLoader(trn_data, batch_size=batch, shuffle=True, drop_last=True)

            #if (fold == 0):
            #    print("len tst_data = ", len(ts_data))
            #    print("len trn_data = ", len(trn_data))
            #    print("len val_data = ", len(val_data))
            #    print()
            #print("Fold: ", fold)

            trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num)

            #average_fold()

            end_time = time.time()
            print("--- Finished Training ---")
            elapsed_time = end_time - start_time
            print("Total Training Time: {:.2f} seconds".format(elapsed_time))
            print()

            plot_training(folds, fold*2 + 1, outc, save_num)


            fold_tst_acc[fold], fold_tst_loss[fold] = test(net, ts_load, outc, save_num, epochs, folds, fold*2 + 1)

        print(("    Cross Validation Avg Acc: {:.5f}, Cross Validation Avg Loss: {:.5f}").format(np.mean(fold_tst_acc), np.mean(fold_tst_loss)))

        end_cv_time = time.time()
        print("--- Finished Cross Validation ---")
        elapsed_cv_time = end_cv_time - start_cv_time
        print("Total Cross Validation Time: {:.2f} seconds".format(elapsed_cv_time))
        print()

        return ts_load


















def train(net, batch, val_per, lrn_rate, mntum, folds, epochs, cross_val, outc, save_num):
    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    torch.save(net.state_dict(), "../untrained_"+outc+"Net")
    
    path = "../"+outc+"Net"+str(save_num)
    print("path = ", path)
    outc_idx = outcome_to_index[outc]
    print("outc = ", outc)
    print("outc_idx = ", outc_idx)

    #print("--- Starting Training ---")
    #start_time = time.time()

    tvs_data = load_data()
    splits = [1/folds] * folds
    tvs_folds = torch.utils.data.random_split(tvs_data, splits)

    print()


    if (cross_val == False):
        print("--- Starting Training ---")
        start_time = time.time()
        fold = random.randrange(folds)

        print("Test Fold: ", fold)

        ts_data = tvs_folds[fold]
        tv_list = tvs_folds.copy()
        tv_list.pop(fold)
        tv_data = torch.utils.data.ConcatDataset(tv_list)

        ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True, drop_last=True)

        tvsplits = len(tv_data)
        val_split = int(tvsplits * val_per)
        trn_split = tvsplits - val_split
        trn_data, val_data = torch.utils.data.random_split(tv_data, [trn_split, val_split])

        val_load = DataLoader(val_data, batch_size=batch, shuffle=True, drop_last=True)
        trn_load = DataLoader(trn_data, batch_size=batch, shuffle=True, drop_last=True)

        #print("len tst_data = ", len(ts_data))
        #print("len trn_data = ", len(trn_data))
        #print("len val_data = ", len(val_data))
        #print()
        #print("Test Fold: ", fold)

        trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num)

        end_time = time.time()
        print("--- Finished Training ---")
        elapsed_time = end_time - start_time
        print("Total Training Time: {:.2f} seconds".format(elapsed_time))
        print()

        plot_training(1, 1, outc, save_num)

        fold_tst_acc, fold_tst_loss = test(net, ts_load, outc, save_num, epochs, 1, 1)

        #average_fold()


        return ts_load

    else:

        print("--- Starting Cross Validation ---")
        start_cv_time = time.time()

        fold_tst_loss = np.zeros(folds)
        fold_tst_acc = np.zeros(folds)
        
        for fold in range(folds):
            net.load_state_dict(torch.load("../untrained_"+outc+"Net"))
            if torch.cuda.is_available():
                net.cuda()
                #print("Using GPU")
            else:
                print("Using CPU")
            print("--- Starting Training ---")
            start_time = time.time()
            print("Fold: ", fold)
            ts_data = tvs_folds[fold]
            tv_list = tvs_folds.copy()
            tv_list.pop(fold)
            tv_data = torch.utils.data.ConcatDataset(tv_list)

            #print("len ts_data = ", len(ts_data))
            #print("len tv_data = ", len(tv_data))

            ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True, drop_last=True)

            tvsplits = len(tv_data)
            val_split = int(tvsplits * val_per)
            trn_split = tvsplits - val_split
            trn_data, val_data = torch.utils.data.random_split(tv_data, [trn_split, val_split])

            val_load = DataLoader(val_data, batch_size=batch, shuffle=True, drop_last=True)
            trn_load = DataLoader(trn_data, batch_size=batch, shuffle=True, drop_last=True)

            #if (fold == 0):
            #    print("len tst_data = ", len(ts_data))
            #    print("len trn_data = ", len(trn_data))
            #    print("len val_data = ", len(val_data))
            #    print()
            #print("Fold: ", fold)

            trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num)

            #average_fold()

            end_time = time.time()
            print("--- Finished Training ---")
            elapsed_time = end_time - start_time
            print("Total Training Time: {:.2f} seconds".format(elapsed_time))
            print()

            plot_training(folds, fold*2 + 1, outc, save_num)


            fold_tst_acc[fold], fold_tst_loss[fold] = test(net, ts_load, outc, save_num, epochs, folds, fold*2 + 1)

        print(("    Cross Validation Avg Acc: {:.5f}, Cross Validation Avg Loss: {:.5f}").format(np.mean(fold_tst_acc), np.mean(fold_tst_loss)))

        end_cv_time = time.time()
        print("--- Finished Cross Validation ---")
        elapsed_cv_time = end_cv_time - start_cv_time
        print("Total Cross Validation Time: {:.2f} seconds".format(elapsed_cv_time))
        print()

        return ts_load


#Tests net (Specified by outc and save_num) on given dataset [ne, nf used for graphing test lines on the train/val graphs]
def test(net, dataset, outc, save_num, ne, num_sbplt, cur_sbplt,):
    print("--- Starting Testing ---")
    path = "../"+outc+"Net"+str(save_num)
    #print("path = ", path)

    outc_idx = outcome_to_index[outc]
    #print("outc = ", outc)
    #print("outc_idx = ", outc_idx)
    if (outc_idx == 3) or (outc_idx == 5):
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    
    if torch.cuda.is_available():
        net.cuda()
        #print("Using GPU")
    else:
        print("Using CPU")

    net.load_state_dict(torch.load(path))
    ts_acc, ts_loss = validate(net, dataset, criterion, outc)
    print(("    Test Acc: {}, Test Loss: {}").format(ts_acc, ts_loss))
    print("--- Finished Testing ---")
    print()
    x_pnts = np.arange(ne)
    ts_acc_pnts = np.repeat(ts_acc, ne)
    ts_loss_pnts = np.repeat(ts_loss, ne)
    plt.subplot(num_sbplt, 2, cur_sbplt)
    plt.plot(x_pnts, ts_acc_pnts, label="Test")
    plt.legend(loc='best')
    plt.subplot(num_sbplt, 2, cur_sbplt + 1)
    plt.plot(x_pnts, ts_loss_pnts, label="Test")
    plt.legend(loc='best')
    return ts_acc, ts_loss


#Below is code using/calling the train and test functions (for debug/model building/running/etc)


#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 2, "BackPain")
#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 10, "EQ_IndexTL12")

#EQ IndexTL12 testing - G:3/24

batch = 10
val_per = 0.2
lr = 0.001
mntum = 0.8
nf = 10
ne = 50
outc = "EQ_IndexTL12"
save_num = 0
ts = train(eqidxtl12_net(), batch, val_per, lr, mntum, nf, ne, True, outc, save_num)
#test(eqidxtl12_net(), ts, outc, save_num, nf, ne)
plt.show()


#ODI Score testing
"""
batch = 10
val_per = 0.2
lr = 0.001
mntum = 0.8
nf = 5
ne = 50
outc = "ODIScore"
save_num = 0
ts = train(odiscore_net(), batch, val_per, lr, mntum, nf, ne, False, outc, save_num)
#test(odiscore_net(), ts, outc, save_num, nf, ne)
plt.show()
"""

#ODI4 Final Testing - G:3/24
"""
batch = 10
val_per = 0.2
lr = 0.001
mntum = 0.8
nf = 5
ne = 100
outc = "ODI4_Final"
save_num = 0
ts = train(recovery_net(), batch, val_per, lr, mntum, nf, ne, False, outc, save_num)
#test(recovery_net(), ts, outc, save_num, nf, ne)
plt.show()
"""

#Back Pain Testing
"""
batch = 10
val_per = 0.2
lr = 0.001
mntum = 0.8
nf = 5
ne = 50
outc = "BackPain"
save_num = 0
ts = train(backpain_net(), batch, val_per, lr, mntum, nf, ne, False, outc, save_num)
#test(backpain_net(), ts, outc, save_num, nf, ne)
plt.show()
"""



"""
ts = train(recovery_net(), 10, 0.1, 0.01, 0.9, 10, 10, "Recovery", 0)
test(recovery_net(), ts, "Recovery", 0, 10, 1)
plt.show()
"""


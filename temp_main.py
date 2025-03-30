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

#Update and Fix with nets (This isn't used probably just delete)
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
def hp_search(net, folds, val_per_min, val_per_max, val_per_itr, batch_min, batch_max, batch_itr, lrn_rate_min, lrn_rate_max, lrn_rate_itr, mntum_min, mntum_max, mntum_itr, epochs_min, epochs_max, epochs_itr, outc, save_num, Trace):
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


    val_per_list = np.linspace(val_per_min, val_per_max, val_per_itr)
    print("val_per_list = ", val_per_list)

    batch_list = np.floor(np.linspace(batch_min, batch_max, batch_itr))
    print("batch_list = ", batch_list)

    lrn_rate_list = np.linspace(lrn_rate_min, lrn_rate_max, lrn_rate_itr)
    print("lrn_rate_list = ", lrn_rate_list)

    mntum_list = np.linspace(mntum_min, mntum_max, mntum_itr)
    print("mntum_list = ", mntum_list)

    epochs_list = np.floor(np.linspace(epochs_min, epochs_max, epochs_itr))
    print("epochs_list = ", epochs_list)

    num_runs = val_per_itr * batch_itr * lrn_rate_itr * mntum_itr * epochs_itr

    print("num_runs = ", num_runs)

    final_list = np.zeros((num_runs, 9))
    #print("final_list = ", final_list)
    #print("final_list.shape = ", final_list.shape)

    print()

    """
    i = 0
    for epochs in epochs_list:
        #print("epochs = ", epochs)
        for batch in batch_list:
            #print("batch = ", batch)
            for lrn_rate in lrn_rate_list:
                #print("lrn_rate = ", lrn_rate)
                for mntum in mntum_list:
                    #print("mntum = ", mntum)
                    for val_per in val_per_list:
                        #print("val_per = ", val_per)
                        i=i+1
    print("i = ", i)
    """

    #exit()
    

    print("--- Starting Hyperparameter Search ---")
    start_hps_time = time.time()

    
    curb_ts_acc = 0
    curb_ts_loss = 0

    curb_score = 0
    curb_epochs = 0
    curb_batch = 0
    curb_lrn_rate = 0
    curb_mntum = 0
    curb_val_per = 0

    i = 0
    for epochs in epochs_list:
        epochs = int(epochs)
        
        for batch in batch_list:
            batch = int(batch)
            
            for lrn_rate in lrn_rate_list:
                
                for mntum in mntum_list:
                    
                    for val_per in val_per_list:
                        net.load_state_dict(torch.load("../untrained_"+outc+"Net"))

                        if Trace:
                            print("--- Starting Training ---")
                        start_time = time.time()
                        fold = random.randrange(folds)

                        if Trace:
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

                        trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num, Trace)

                        end_time = time.time()
                        if Trace:
                            print("--- Finished Training ---")
                        elapsed_time = end_time - start_time
                        if Trace:
                            print("Total Training Time: {:.2f} seconds".format(elapsed_time))
                            print()

                        plot_training(1, 1, outc, save_num)

                        fold_tst_acc, fold_tst_loss = test(net, ts_load, outc, save_num, epochs, 1, 1, False, Trace)

                        cur_score = fold_tst_acc + (1/(1 + fold_tst_loss))

                        if Trace:
                            print(("    Cur Score: {:.5f}, Cur Test Acc: {:.5f}, Cur Test Loss: {:.5f}").format(cur_score, fold_tst_acc, fold_tst_loss))
                            print(("    Cur Epoch: {}, Cur Batch: {}, Cur Lrn Rate: {}, Cur Mntum: {}, Cur Val Per: {}").format(epochs, batch, lrn_rate, mntum, val_per))
                            print()


                        final_list[i][0] = fold
                        final_list[i][1] = cur_score
                        final_list[i][2] = fold_tst_acc
                        final_list[i][3] = fold_tst_loss
                        final_list[i][4] = epochs
                        final_list[i][5] = batch
                        final_list[i][6] = lrn_rate
                        final_list[i][7] = mntum
                        final_list[i][8] = val_per
                        if not Trace:
                            print(("    Test Fold: {}, Test Score: {:.5f}, Test Acc: {:.5f}, Test Loss: {:.5f} | "+"Epochs: {}, Batch: {}, Lrn Rate: {:.4f}, Mntum: {}, Val Per: {}").format(final_list[i][0], final_list[i][1], final_list[i][2], final_list[i][3], final_list[i][4], final_list[i][5], final_list[i][6], final_list[i][7], final_list[i][8]))
                        i=i+1
 

                        if cur_score > curb_score:
                            curb_score = cur_score
                            curb_ts_acc = fold_tst_acc
                            curb_ts_loss = fold_tst_loss
                            curb_epochs = epochs
                            curb_batch = batch
                            curb_lrn_rate = lrn_rate
                            curb_mntum = mntum
                            curb_val_per = val_per

    if Trace:
        for j in range(num_runs):    
            print(("    Test Fold: {}, Test Score: {:.5f}, Test Acc: {:.5f}, Test Loss: {:.5f} | "+"Epochs: {}, Batch: {}, Lrn Rate: {:.4f}, Mntum: {}, Val Per: {}").format(final_list[j][0], final_list[j][1], final_list[j][2], final_list[j][3], final_list[j][4], final_list[j][5], final_list[j][6], final_list[j][7], final_list[j][8]))
        print()
    
    if not Trace:
        print()
    print(("    Best Score: {:.5f}, Best Test Acc: {:.5f}, Best Test Loss: {:.5f}").format(curb_score, curb_ts_acc, curb_ts_loss))
    print(("    Best Epoch: {}, Best Batch: {}, Best Lrn Rate: {:.4f}, Best Mntum: {}, Best Val Per: {}").format(curb_epochs, curb_batch, curb_lrn_rate, curb_mntum, curb_val_per))
    print("--- Finished Hyperparameter Search ---")
    end_hps_time = time.time()
    elapsed_hps_time = end_hps_time - start_hps_time
    print("Total Hyperparameter Search Time: {:.2f} seconds".format(elapsed_hps_time))
    print()

    return curb_ts_acc, curb_ts_loss, curb_epochs, curb_batch, curb_lrn_rate, curb_mntum, curb_val_per
    



















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

        trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num, True)

        end_time = time.time()
        print("--- Finished Training ---")
        elapsed_time = end_time - start_time
        print("Total Training Time: {:.2f} seconds".format(elapsed_time))
        print()

        plot_training(1, 1, outc, save_num)

        fold_tst_acc, fold_tst_loss = test(net, ts_load, outc, save_num, epochs, 1, 1, True, True)

        #average_fold()


        return ts_load

    else:

        print("--- Starting Cross Validation ---")
        start_cv_time = time.time()

        fold_tst_loss = np.zeros(folds)
        fold_tst_acc = np.zeros(folds)
        
        for fold in range(folds):
            net.load_state_dict(torch.load("../untrained_"+outc+"Net"))

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

            trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num, False)

            #average_fold()

            end_time = time.time()
            print("--- Finished Training ---")
            elapsed_time = end_time - start_time
            print("Total Training Time: {:.2f} seconds".format(elapsed_time))
            print()

            plot_training(folds, fold*2 + 1, outc, save_num)


            fold_tst_acc[fold], fold_tst_loss[fold] = test(net, ts_load, outc, save_num, epochs, folds, fold*2 + 1, True, True)

        print(("    Cross Validation Avg Acc: {:.5f}, Cross Validation Avg Loss: {:.5f}").format(np.mean(fold_tst_acc), np.mean(fold_tst_loss)))

        end_cv_time = time.time()
        print("--- Finished Cross Validation ---")
        elapsed_cv_time = end_cv_time - start_cv_time
        print("Total Cross Validation Time: {:.2f} seconds".format(elapsed_cv_time))
        print()

        return ts_load


#Tests net (Specified by outc and save_num) on given dataset [ne, nf used for graphing test lines on the train/val graphs]
def test(net, dataset, outc, save_num, ne, num_sbplt, cur_sbplt, plot, Trace):
    if Trace:
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
    if Trace:
        print(("    Test Acc: {:.5f}, Test Loss: {:.5f}").format(ts_acc, ts_loss))
        print("--- Finished Testing ---")
        print()

    if plot:
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

#Flow will be: call hp_search, use best hp with cross_validate=True, if cv okay use best hp with cross_validate=False, use returned ts_data and saved net for net_analysis

#EQ IndexTL12 testing - G:3/28

batch = 10
val_per = 0.2
lr = 0.001
mntum = 0.8
nf = 10
ne = 30
outc = "EQ_IndexTL12"
save_num = 0
#ts = train(eqidxtl12_net(), batch, val_per, lr, mntum, nf, ne, False, outc, save_num)
#test(eqidxtl12_net(), ts, outc, save_num, nf, ne)
#plt.show()

val_per_min = 0.2
val_per_max = 0.2
val_per_itr = 1
batch_min = 2
batch_max = 32
batch_itr = 4
lrn_rate_min = 0.001 
lrn_rate_max = 0.1
lrn_rate_itr = 1
mntum_min = 0.8
mntum_max = 0.9
mntum_itr = 1
epochs_min = 30
epochs_max = 30
epochs_itr = 1

hp_search(eqidxtl12_net(), nf, val_per_min, val_per_max, val_per_itr, batch_min, batch_max, batch_itr, lrn_rate_min, lrn_rate_max, lrn_rate_itr, mntum_min, mntum_max, mntum_itr, epochs_min, epochs_max, epochs_itr, outc, save_num, True)


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
nf = 10
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


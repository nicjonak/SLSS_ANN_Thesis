#Temp Main
#Temporary Main file, contains train and test functions that intiate, visualize, and time training and testing

import time
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from test_nets import *
from net_trainer import *
from data_visualizer import *

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


#Below is code using/calling the train and test functions (for debug/model building/running/etc)


#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 2, "BackPain")
#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 10, "EQ_IndexTL12")

#EQ IndexTL12 testing

batch = 10
ts_per = 0.1
lr = 0.01
mntum = 0.9
nf = 10
ne = 5
outc = "EQ_IndexTL12"
save_num = 0
ts = train(eqidxtl12_net(), batch, ts_per, lr, mntum, nf, ne, outc, save_num)
test(eqidxtl12_net(), ts, outc, save_num, nf, ne)
plt.show()


#ODI Score testing
"""
batch = 10
ts_per = 0.1
lr = 0.01
mntum = 0.9
nf = 10
ne = 3
outc = "ODIScore"
save_num = 0
ts = train(odiscore_net(), batch, ts_per, lr, mntum, nf, ne, outc, save_num)
test(odiscore_net(), ts, outc, save_num, nf, ne)
plt.show()
"""

#ODI4 Final Testing
"""
batch = 10
ts_per = 0.1
lr = 0.01
mntum = 0.9
nf = 10
ne = 5
outc = "ODI4_Final"
save_num = 0
ts = train(recovery_net(), batch, ts_per, lr, mntum, nf, ne, outc, save_num)
test(recovery_net(), ts, outc, save_num, nf, ne)
plt.show()
"""

#Back Pain Testing
"""
batch = 10
ts_per = 0.1
lr = 0.01
mntum = 0.9
nf = 10
ne = 10
outc = "BackPain"
save_num = 0
ts = train(backpain_net(), batch, ts_per, lr, mntum, nf, ne, outc, save_num)
test(backpain_net(), ts, outc, save_num, nf, ne)
plt.show()
"""



"""
ts = train(recovery_net(), 10, 0.1, 0.01, 0.9, 10, 10, "Recovery", 0)
test(recovery_net(), ts, "Recovery", 0, 10, 1)
plt.show()
"""


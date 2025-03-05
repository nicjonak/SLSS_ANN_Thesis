import time
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from test_nets import *
from net_trainer import *

def train(net, batch, ts_per, lrn_rate, mntum, folds, epochs, outc):
    print("--- Starting Training ---")
    start_time = time.time()
    tv_data, ts_data = load_data(ts_per)
    ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True)

    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    trainNet(tv_data, net, batch, lrn_rate, mntum, folds, epochs, outc)
    end_time = time.time()
    print("--- Finished Training ---")
    elapsed_time = end_time - start_time
    print("Total Training Time: {:.2f} seconds".format(elapsed_time))
    return ts_load

#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 2, "BackPain")
#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 10, "EQ_IndexTL12")
"""
print("")
print("         --- Break ---")
print("")
"""
train(recovery_net(), 10, 0.1, 0.0005, 0.8, 10, 25, "Recovery")
"""
print("")
print("         --- Break ---")
print("")

train(smplNetLog(), 10, 0.1, 0.01, 0.5, 10, 2, "ODI4_Final")
"""
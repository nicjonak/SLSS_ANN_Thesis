import time
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from test_nets import *
from net_trainer import *

def train(net, batch, ts_per, lrn_rate, mntum, folds):
    print("--- Starting Training ---")
    start_time = time.time()
    tv_data, ts_data = load_data(ts_per)
    ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True)

    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    trainNet(tv_data, net, batch, lrn_rate, mntum, folds)
    end_time = time.time()
    print("--- Finished Training ---")
    elapsed_time = end_time - start_time
    print("Total Training Time: {:.2f} seconds".format(elapsed_time))
    return ts_load

train(smplNet(), 10, 0.1, 0.05, 0.9, 10)

import time
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import *
from test_nets import *
from net_trainer import *
from data_visualizer import *

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
    plot_training(True, outc, save_num)
    return ts_load


def test(net, dataset, outc, save_num):
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


#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 2, "BackPain")
#train(smplNetCnt(), 10, 0.1, 0.001, 0.1, 10, 10, "EQ_IndexTL12")
"""
print("")
print("         --- Break ---")
print("")
"""
#ts = train(recovery_net(), 10, 0.1, 0.0005, 0.8, 10, 3, "Recovery", 0)

ts = train(recovery_net(), 10, 0.1, 0.0001, 0.6, 10, 100, "Recovery", 0)
#print("         ----- GETS HERE -----")
#train(recovery_net(), 10, 0.1, 0.0005, 0.8, 5, 2, "Recovery", 0)
#print("         ----- GETS HERE 2 -----")
test(recovery_net(), ts, "Recovery", 0)
plt.show()
"""
print("")
print("         --- Break ---")
print("")

train(smplNetLog(), 10, 0.1, 0.01, 0.5, 10, 2, "ODI4_Final")
"""
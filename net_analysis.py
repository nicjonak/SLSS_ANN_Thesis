#Net Analysis
#Functions to analyze/evaluate trained nets

import torch
from net_trainer import *
from data_loader import * #temp for debugging
from test_nets import * #temp for debugging

def evaluate_net(net, dataset, outc, save_num):
    print("--- Starting Evaluation ---")
    path = "../"+outc+"Net"+str(save_num)
    #print("path = ", path)

    outc_idx = outcome_to_index[outc]
    #print("outc = ", outc)
    #print("outc_idx = ", outc_idx)
    
    if torch.cuda.is_available():
        net.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    net.load_state_dict(torch.load(path))

    #Either add break or counter to limit from single data point to entire test set used and averaged to find the critical predictors
    for i, data in enumerate(dataset):
        inputs = data['predictors']
        outcomes = data['outcomes']
        outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
        if (outc_idx == 3) or (outc_idx == 5):
            outcomes = Log_Convert(outcomes, outc)

        print(" Outcomes.size = ", outcomes.size())
        print(" Outcomes = ", outcomes)

        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        outputs = net(inputs)

        print(" Output.size = ", outputs.size())
        print(" Output = ", outputs)

        
#temp for debugging
"""
tv_data, ts_data = load_data(0.1)
ts_load = DataLoader(ts_data, batch_size=10, shuffle=True)
outc = "EQ_IndexTL12"
save_num = 0
evaluate_net(eqidxtl12_net(), ts_load, outc, save_num)
"""
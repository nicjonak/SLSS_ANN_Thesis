#Net Trainer
#Functions to Train and Validate nets along with Helper funcs 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

outcome_to_index = {
        "BackPain": 0,
        "LegPain": 1,
        "ODIScore": 2,
        "ODI4_Final": 3,
        "EQ_IndexTL12": 4,
        "Recovery": 5,
        "Full": 6
        }

index_to_outcome = {
        0: "BackPain",
        1: "LegPain",
        2: "ODIScore",
        3: "ODI4_Final",
        4: "EQ_IndexTL12",
        5: "Recovery",
        6: "Full"
        }

outcome_to_range = {
        "BackPain": 10,
        "LegPain": 10,
        "ODIScore": 100,
        "ODI4_Final": 2,
        "EQ_IndexTL12": 1,
        "Recovery": 1
        }

#Convert categorical outcomes to One-Hot outcome vectors
def Log_Convert(outcomes, outc):
    outc_idx = outcome_to_index[outc]
    out1 = int(outcomes[0,0].item())
    """
    if (outc_idx == 3):
        if out1 == 1:
            out1 = 0
        elif out1 == 2:
            out1 = 1
        else:
            print("ERROR: ODI4 value other than 1 or 2 shouldn't happen")
    """
    tmp1 = torch.zeros(1,2)
    tmp1[0,out1] = 1
    ret = tmp1
    
    for i in range(1,len(outcomes)):
        out_val = int(outcomes[i,0].item())
        if (outc_idx == 3):
            if out_val == 1:
                out_val = 0
            elif out_val == 2:
                out_val = 1
            else:
                print("ERROR: ODI4 value other than 1 or 2 shouldn't happen")

        tmp_enc = torch.zeros(1,2)
        tmp_enc[0,out_val] = 1
        ret = torch.cat((ret,tmp_enc), 0)
    return ret

#Convert net outputs for comparison, Currently written for categorical outcomes to round probability vector to prediction vector
def convert_outputs(outputs, outc):
    outc_idx = outcome_to_index[outc]
    ret = outputs.clone().detach()
    for i in range(0,len(outputs)):
        ret[i] = torch.round(ret[i])
    return ret

#Round net outputs for comparison, Currently written for continuous outcomes to round decimal continous outcomes to nearest dichotomized box
def round_outputs(outputs, outc):
    outc_idx = outcome_to_index[outc]
    ret = outputs.clone().detach()
    for i in range(0,len(outputs)):
        ret[i] = torch.round(ret[i])
    #print("ret = ", ret)
    return ret

#Calculate number of correct net predicted outputs against true outcomes
def calc_cor(outputs, outcomes):
    total_cor = 0
    if len(outputs) == len(outcomes):
        for i in range(0,len(outputs)):
            #print("i =", i)
            #print("outcomes[i] = ", outcomes[i])
            #print("outputs[i] = ", outputs[i])

            if torch.equal(outputs[i], outcomes[i]):
                total_cor+=1
    else:
        print("ERROR: calc_cor(): outputs and outcomes different size")
        return 0.0
    return total_cor

#Calculate error between net predicted outputs and true outcomes
def calc_err(outputs, outcomes, outcN):
    total_err = 0
    if len(outputs) == len(outcomes):
        for i in range(0,len(outputs)):
            outp = outputs[i].item()
            outc = outcomes[i].item()
            #print("i =", i)
            #print("outcomes[",i,"] = ", outc)
            #print("outputs[",i,"] = ", outp)
            #print("err[",i,"] = ", abs(outp - outc))

            total_err+= abs(outp - outc) / outcome_to_range[outcN]
            
    else:
        print("ERROR: calc_err(): outputs and outcomes different size")
        return 0.0
    #print("total_err = ", total_err)
    return total_err




#Calculate cor/error for Full net/outcomes
def calc_corerr(outputs, outcomes, outcN):
    total_err = 0
    if len(outputs) == len(outcomes):
        for i in range(0,len(outputs)):
            temp_err = 0.0
            for j in range(0,len(outputs[i])):
                outp = outputs[i,j].item()
                outc = outcomes[i,j].item()
                #if j == 3:
                #    outp = np.round(outp) * 2
                if j == 5:
                    outp = np.round(outp)
            #print("i =", i)
            #print("outcomes[",i,"] = ", outc)
            #print("outputs[",i,"] = ", outp)
            #print("err[",i,"] = ", abs(outp - outc))

                temp_err+= abs(outp - outc) / outcome_to_range[index_to_outcome[j]]
            total_err+= temp_err/6
            
    else:
        print("ERROR: calc_corerr(): outputs and outcomes different size")
        return 0.0
    #print("total_err = ", total_err)
    return total_err





#Validate net over given dataset with given criterion
def validate(net, dataset, criterion, outc):
    outc_idx = outcome_to_index[outc]
    total_loss = 0.0
    total_acc = 0.0
    total_data = 0
    for i, data in enumerate(dataset):
        inputs = data['predictors']
        outcomes = data['outcomes']
        if outc_idx != 6:
            outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
            #if (outc_idx == 5): # or (outc_idx == 3):
            #    outcomes = Log_Convert(outcomes, outc)

        #print("    --- Start ---")
        #print(" Inputs.size = ", inputs.size())
        #print(" Inputs = ", inputs)
        #print(" Outcomes.size = ", outcomes.size())
        #print(" Outcomes = ", outcomes)

        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        outputs = net(inputs)
        #print(" Output = ", outputs)
        #print("    --- End ---")
        loss = criterion(outputs, outcomes)

        if (outc_idx == 5): # or (outc_idx == 3):
            conv_outp = convert_outputs(outputs, outc)
            cur_cor = calc_cor(conv_outp, outcomes)
        elif (outc_idx == 6):
                cur_cor = calc_corerr(outputs, outcomes, outc)
        else:
            cur_cor = calc_err(outputs, outcomes, outc)

        total_acc += cur_cor
        total_loss += loss.item()
        total_data += len(outcomes)

    if (outc_idx == 5): # or (outc_idx == 3):
        acc = float(total_acc) / total_data
    else:
        acc = 1 - (float(total_acc) / total_data)

    loss = float(total_loss) / (i+1)
    return acc, loss


def trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num, Trace):
    path = "../"+outc+"Net"+str(save_num)
    #print("path = ", path)
    outc_idx = outcome_to_index[outc]
    #print("outc = ", outc)
    #print("outc_idx = ", outc_idx)
    #print()
    if (outc_idx == 5): # or (outc_idx == 3):
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
    #elif (outc_idx == 2) or (outc_idx == 1) or (outc_idx == 0) or (outc_idx == 6):
    #    criterion = nn.L1Loss()
    else:
        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=lrn_rate, momentum=mntum)
    #optimizer = optim.Adam(net.parameters(), lr=lrn_rate)


    trn_acc = np.zeros(epochs)
    trn_loss = np.zeros(epochs)
    val_acc = np.zeros(epochs)
    val_loss = np.zeros(epochs)

    for epoch in range(epochs):

        ep_trn_loss = 0.0
        ep_trn_acc = 0.0
        ep_data = 0

        for i, data in enumerate(trn_load):
            inputs = data['predictors']
            outcomes = data['outcomes']
            #outc_idx = outcome_to_index[outc]
            if outc_idx != 6:
                outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
                #if (outc_idx == 5): # or (outc_idx == 3):
                #    outcomes = Log_Convert(outcomes, outc)
            else:
                outcomes = outcomes.to(torch.float32)
            #print("    --- Start ---")
            #print(" Outcomes.size = ", outcomes.size())
            #print(" Outcomes = ", outcomes)
            #print(" Outcomes[BackPain] = ", outcomes[:,0])
            #return

            if torch.cuda.is_available() == True:
                inputs = inputs.cuda()
                outcomes = outcomes.cuda()
                #print("Using cuda")
            else:
                print("Using CPU")

            optimizer.zero_grad()
            outputs = net(inputs)
            #print(" Output.size = ", outputs.size())
            #print(" Output = ", outputs)
            #print(" -- Gets Here -- ")
            loss = criterion(outputs, outcomes)
            loss.backward()
            optimizer.step()
            
            #print("  loss.item() = ",loss.item())
            if (outc_idx == 5): # or (outc_idx == 3):
                conv_outp = convert_outputs(outputs, outc)
                #print("post conv outputs = ", outputs)
                cur_cor = calc_cor(conv_outp, outcomes)
            elif (outc_idx == 6):
                cur_cor = calc_corerr(outputs, outcomes, outc)
            else:
                cur_cor = calc_err(outputs, outcomes, outc)

            #print("      cur_cor = ", cur_cor)
            ep_trn_acc += cur_cor
            #print("      ep_train_acc = ", ep_trn_acc)
            ep_trn_loss += loss.item()
            ep_data += len(outcomes)
            #print("      ep_data = ", ep_data)
            #print("    --- End ---")

        if (outc_idx == 5): # or (outc_idx == 3):
            trn_acc[epoch] = float(ep_trn_acc) / ep_data
        else:
            trn_acc[epoch] = 1 - (float(ep_trn_acc) / ep_data)

        trn_loss[epoch] = float(ep_trn_loss) / (i+1)
        val_acc[epoch], val_loss[epoch] = validate(net, val_load, criterion, outc)
        if Trace:
            print(("    Epoch {}: Train Acc: {:.5f}, Train Loss: {:.5f} | "+"Val Acc: {:.5f}, Val Loss: {:.5f}").format(epoch, trn_acc[epoch], trn_loss[epoch], val_acc[epoch], val_loss[epoch]))

    
    torch.save(net.state_dict(), path)
    np.savetxt("{}_train_acc.csv".format(path), trn_acc)
    np.savetxt("{}_train_loss.csv".format(path), trn_loss)
    np.savetxt("{}_val_acc.csv".format(path), val_acc)
    np.savetxt("{}_val_loss.csv".format(path), val_loss)
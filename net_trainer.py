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
        "Recovery": 5
        }

#Convert categorical outcomes to One-Hot outcome vectors
def Log_Convert(outcomes, outc):
    outc_idx = outcome_to_index[outc]
    out1 = int(outcomes[0,0].item())
    if (outc_idx == 3):
        if out1 == 1:
            out1 = 0
        elif out1 == 2:
            out1 = 1
        else:
            print("ERROR: ODI4 value other than 1 or 2 shouldn't happen")
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
            print("i =", i)
            print("outcomes[i] = ", outcomes[i])
            print("outputs[i] = ", outputs[i])

            if torch.equal(outputs[i], outcomes[i]):
                total_cor+=1
    else:
        print("ERROR: calc_cor(): outputs and outcomes different size")
        return 0.0
    return total_cor

#Calculate error between net predicted outputs and true outcomes
def calc_err(outputs, outcomes):
    total_err = 0
    if len(outputs) == len(outcomes):
        for i in range(0,len(outputs)):
            outp = outputs[i].item()
            outc = outcomes[i].item()
            #print("i =", i)
            #print("outcomes[",i,"] = ", outc)
            #print("outputs[",i,"] = ", outp)
            #print("err[",i,"] = ", abs(outp - outc))

            total_err+= abs(outp - outc)
            
    else:
        print("ERROR: calc_err(): outputs and outcomes different size")
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
        outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
        if (outc_idx == 3) or (outc_idx == 5):
            outcomes = Log_Convert(outcomes, outc)

        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        outputs = net(inputs)
        loss = criterion(outputs, outcomes)

        if (outc_idx == 3) or (outc_idx == 5):
            conv_outp = convert_outputs(outputs, outc)
            cur_cor = calc_cor(conv_outp, outcomes)
        elif (outc_idx == 4): #if using EQ IdxTL12 calculate error of output rather than number of correct predictions for accuracy
            cur_cor = calc_err(outputs, outcomes)
        else:
            round_outp = round_outputs(outputs, outc)
            cur_cor = calc_cor(round_outp, outcomes)
        total_acc += cur_cor
        total_loss += loss.item()
        total_data += len(outcomes)
    if (outc_idx == 4): #if using EQ IdxTL12 use error of output to calculate accuracy
        acc = 1 - (float(total_acc) / total_data)
    else:
        acc = float(total_acc) / total_data
    loss = float(total_loss) / (i+1)
    return acc, loss

#Train net using given hyperparameters and data, save final net, acc, and loss into .csv files
def trainNet(tv_data, net, batch, lrn_rate, mntum, folds, epochs, outc, save_num):
    path = "../"+outc+"Net"+str(save_num)
    print("path = ", path)
    outc_idx = outcome_to_index[outc]
    print("outc = ", outc)
    print("outc_idx = ", outc_idx)
    if (outc_idx == 3) or (outc_idx == 5):
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    #optimizer = optim.SGD(net.parameters(), lr=lrn_rate, momentum=mntum)
    optimizer = optim.Adam(net.parameters(), lr=lrn_rate)

    ep_trn_acc = np.zeros(epochs)
    ep_trn_loss = np.zeros(epochs)
    ep_val_acc = np.zeros(epochs)
    ep_val_loss = np.zeros(epochs)

    trn_acc = np.zeros((epochs, folds))
    trn_loss = np.zeros((epochs, folds))
    val_acc = np.zeros((epochs, folds))
    val_loss = np.zeros((epochs, folds))

    for epoch in range(epochs):
        #trn_acc = np.zeros(folds)
        #trn_loss = np.zeros(folds)
        #val_acc = np.zeros(folds)
        #val_loss = np.zeros(folds)

        #for epoch in range(epochs)
        splits = [1/folds] * folds
        tv_folds = torch.utils.data.random_split(tv_data, splits)
        print("Epoch: ",epoch + 1)
        for fold in range(folds):
            val_data = tv_folds[fold]
            trn_list = tv_folds.copy()
            trn_list.pop(fold)
            trn_data = torch.utils.data.ConcatDataset(trn_list)
            val_load = DataLoader(val_data, batch_size=batch, shuffle=True)
            trn_load = DataLoader(trn_data, batch_size=batch, shuffle=True)

            fold_trn_loss = 0.0
            fold_trn_acc = 0.0
            fold_data = 0

            for i, data in enumerate(trn_load):
                inputs = data['predictors']
                outcomes = data['outcomes']
                #outc_idx = outcome_to_index[outc]
                outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
                if (outc_idx == 3) or (outc_idx == 5):
                    outcomes = Log_Convert(outcomes, outc)
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
                if (outc_idx == 3) or (outc_idx == 5):
                    conv_outp = convert_outputs(outputs, outc)
                    #print("post conv outputs = ", outputs)
                    cur_cor = calc_cor(conv_outp, outcomes)
                elif (outc_idx == 4): #if using EQ IdxTL12 calculate error of output rather than number of correct predictions for accuracy
                    cur_cor = calc_err(outputs, outcomes)
                else:
                    round_outp = round_outputs(outputs, outc)
                    #print("post round outputs = ", outputs)
                    cur_cor = calc_cor(round_outp, outcomes)
                #print("      cur_cor = ", cur_cor)
                fold_trn_acc += cur_cor
                #print("      fold_train_acc = ", fold_trn_acc)
                #print("    --- End ---")
                fold_trn_loss += loss.item()
                fold_data += len(outcomes)
            if (outc_idx == 4): #if using EQ IdxTL12 use error of output to calculate accuracy
                trn_acc[epoch, fold] = 1 - (float(fold_trn_acc) / fold_data)
            else:
                trn_acc[epoch, fold] = float(fold_trn_acc) / fold_data
            trn_loss[epoch, fold] = float(fold_trn_loss) / (i+1)
            val_acc[epoch, fold], val_loss[epoch, fold] = validate(net, val_load, criterion, outc)
            print(("    Fold {}: Train Acc: {:.5f}, Train Loss: {:.5f} | "+"Val Acc: {:.5f}, Val Loss: {:.5f}").format(fold + 1, trn_acc[epoch, fold], trn_loss[epoch, fold], val_acc[epoch, fold], val_loss[epoch, fold]))
        ep_trn_acc[epoch] = np.mean(trn_acc[epoch])
        ep_trn_loss[epoch] = np.mean(trn_loss[epoch])
        ep_val_acc[epoch] = np.mean(val_acc[epoch])
        ep_val_loss[epoch] = np.mean(val_loss[epoch])
        print(("  Epoch {}: Avg Train Acc: {:.5f}, Avg Train Loss: {:.5f} | "+"Avg Val Acc: {:.5f}, Avg Val Loss: {:.5f}").format(epoch + 1, ep_trn_acc[epoch], ep_trn_loss[epoch], ep_val_acc[epoch], ep_val_loss[epoch]))
    
    torch.save(net.state_dict(), path)
    np.savetxt("{}_train_acc.csv".format(path), trn_acc)
    np.savetxt("{}_train_loss.csv".format(path), trn_loss)
    np.savetxt("{}_val_acc.csv".format(path), val_acc)
    np.savetxt("{}_val_loss.csv".format(path), val_loss)

    np.savetxt("{}_ep_train_acc.csv".format(path), ep_trn_acc)
    np.savetxt("{}_ep_train_loss.csv".format(path), ep_trn_loss)
    np.savetxt("{}_ep_val_acc.csv".format(path), ep_val_acc)
    np.savetxt("{}_ep_val_loss.csv".format(path), ep_val_loss)
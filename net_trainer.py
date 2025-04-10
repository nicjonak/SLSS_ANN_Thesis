#Net Trainer
#Functions to Train and Validate nets along with Helper funcs 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Dict to convert outcome strings to their index within dataset
outcome_to_index = {
        "BackPain": 0,
        "LegPain": 1,
        "ODIScore": 2,
        "ODI4_Final": 3,
        "EQ_IndexTL12": 4,
        "Recovery": 5,
        "Full": 6
        }

#Dict to convert indices to their outcome strings
index_to_outcome = {
        0: "BackPain",
        1: "LegPain",
        2: "ODIScore",
        3: "ODI4_Final",
        4: "EQ_IndexTL12",
        5: "Recovery",
        6: "Full"
        }

#Dict to convert outcome strings to their data ranges
outcome_to_range = {
        "BackPain": 10,
        "LegPain": 10,
        "ODIScore": 100,
        "ODI4_Final": 2,
        "EQ_IndexTL12": 1,
        "Recovery": 1
        }

#Convert net outputs for comparison, written for categorical outcomes to round probability vector to prediction vector
def convert_outputs(outputs):
    ret = outputs.clone().detach()
    for i in range(0,len(outputs)):
        ret[i] = torch.round(ret[i])
    return ret

#Calculate number of correct net predicted outputs against true outcomes
def calc_cor(outputs, outcomes):
    total_cor = 0
    if len(outputs) == len(outcomes):
        for i in range(0,len(outputs)):
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

            total_err+= abs(outp - outc) / outcome_to_range[outcN]
    else:
        print("ERROR: calc_err(): outputs and outcomes different size")
        return 0.0

    return total_err

#Calculate cor/error for Full net/outcomes
def calc_corerr(outputs, outcomes):
    total_err = 0
    if len(outputs) == len(outcomes):
        for i in range(0,len(outputs)):
            temp_err = 0.0
            for j in range(0,len(outputs[i])):
                outp = outputs[i,j].item()
                outc = outcomes[i,j].item()
                if j == 5:
                    outp = np.round(outp)

                temp_err+= abs(outp - outc) / outcome_to_range[index_to_outcome[j]]
            total_err+= temp_err/6
    else:
        print("ERROR: calc_corerr(): outputs and outcomes different size")
        return 0.0

    return total_err

#Validate net over given dataset with given criterion, output the Avg Val Accuracy and Loss (simply runs data through net, used for testing too)
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

        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        outputs = net(inputs)
        loss = criterion(outputs, outcomes)

        if (outc_idx == 5):
            conv_outp = convert_outputs(outputs)
            cur_cor = calc_cor(conv_outp, outcomes)
        elif (outc_idx == 6):
                cur_cor = calc_corerr(outputs, outcomes)
        else:
            cur_cor = calc_err(outputs, outcomes, outc)

        total_acc += cur_cor
        total_loss += loss.item()
        total_data += len(outcomes)

    if (outc_idx == 5):
        acc = float(total_acc) / total_data
    else:
        acc = 1 - (float(total_acc) / total_data)

    loss = float(total_loss) / (i+1)
    return acc, loss

#trainNet function trains given net using given data and hyperparameters then saves net state and training stats
def trainNet(trn_load, val_load, net, batch, lrn_rate, mntum, epochs, outc, save_num, Trace):
    path = "../"+outc+"Net"+str(save_num)
    outc_idx = outcome_to_index[outc]

    if (outc_idx == 5):
        criterion = nn.BCELoss() #If Outcome Recovery use Binary Cross Entropy Loss
    else:
        criterion = nn.L1Loss() #Other outcomes use L1/MAE Loss

    optimizer = optim.SGD(net.parameters(), lr=lrn_rate, momentum=mntum) #Stochastic Gradient Descent for optimizer

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

            if outc_idx != 6:
                outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
            else:
                outcomes = outcomes.to(torch.float32)

            if torch.cuda.is_available() == True:
                inputs = inputs.cuda()
                outcomes = outcomes.cuda()
                #print("Using cuda")
            else:
                print("Using CPU")

            optimizer.zero_grad()
            outputs = net(inputs) 
            loss = criterion(outputs, outcomes)
            loss.backward()
            optimizer.step()
            
            if (outc_idx == 5): #If Outcome Recovery convert probability outputs to category and calc correct
                conv_outp = convert_outputs(outputs)
                cur_cor = calc_cor(conv_outp, outcomes)
            elif (outc_idx == 6): #If All Outcomes use composite error/correct measure
                cur_cor = calc_corerr(outputs, outcomes)
            else: #Other Outcomes calculate error
                cur_cor = calc_err(outputs, outcomes, outc)

            ep_trn_acc += cur_cor
            ep_trn_loss += loss.item()
            ep_data += len(outcomes)

        if (outc_idx == 5):
            trn_acc[epoch] = float(ep_trn_acc) / ep_data #If Outcome Recovery record Avg Accuracy
        else:
            trn_acc[epoch] = 1 - (float(ep_trn_acc) / ep_data) #Other Outcomes calc Avg Accuracy from Avg Error

        trn_loss[epoch] = float(ep_trn_loss) / (i+1) #Record Avg Train Loss
        val_acc[epoch], val_loss[epoch] = validate(net, val_load, criterion, outc) #Validate and record Avg Loss and Avg Acc
        if Trace:
            print(("    Epoch {}: Train Acc: {:.5f}, Train Loss: {:.5f} | "+"Val Acc: {:.5f}, Val Loss: {:.5f}").format(epoch, trn_acc[epoch], trn_loss[epoch], val_acc[epoch], val_loss[epoch]))

    #Save net state and train/val acc/loss
    torch.save(net.state_dict(), path)
    np.savetxt("{}_train_acc.csv".format(path), trn_acc)
    np.savetxt("{}_train_loss.csv".format(path), trn_loss)
    np.savetxt("{}_val_acc.csv".format(path), val_acc)
    np.savetxt("{}_val_loss.csv".format(path), val_loss)
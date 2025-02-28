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




def BackPain_Convert(outcomes):
    print("  Outcomes.size = ", outcomes.size())
    print("  Outcomes = ", outcomes)
    out1 = int(outcomes[0,0].item())
    tmp1 = torch.zeros(1,11)
    tmp1[0,out1] = 1
    ret = tmp1
    print("  RET = ", ret)
    for i in range(1,len(outcomes)):
        print("        i = ", i)
        out_val = int(outcomes[i,0].item())
        print("  out_val = ", out_val)
        tmp_enc = torch.zeros(1,11)
        tmp_enc[0,out_val] = 1
        print("  tmp_enc = ", tmp_enc)
        ret = torch.cat((ret,tmp_enc), 0)
        print("      ret = ", ret)
        #outcomes[i,0] = tmp_enc
        #print(" outcomes = ",outcomes)
    #return outcomes








def calc_cor(outputs, outcomes):
    total_cor = 0
    if len(outputs) == len(outcomes):
        #print(" -- Gets In Calc_Cor --")
        for i in range(0,len(outputs)):

            if torch.equal(outputs[i], outcomes[i]):
                total_cor+=1
    else:
        print("ERROR: calc_cor(): outputs and outcomes different size")
        return 0.0
    return total_cor

def validate(net, dataset, criterion):
    total_loss = 0.0
    total_acc = 0.0
    total_data = 0
    for i, data in enumerate(dataset):
        inputs = data['predictors']
        outcomes = data['outcomes']
        outcomes = outcomes[:,0].unsqueeze(1)

        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        outputs = net(inputs)
        loss = criterion(outputs, outcomes)

        cur_cor = calc_cor(outputs, outcomes)
        total_acc += cur_cor
        total_loss += loss.item()
        total_data += len(outcomes)
    acc = float(total_acc) / total_data
    loss = float(total_loss) / (i+1)
    return acc, loss

def trainNet(tv_data, net, batch, lrn_rate, mntum, folds, epochs, outc):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lrn_rate, momentum=mntum)

    #for epoch in range(epochs):
    trn_acc = np.zeros(folds)
    trn_loss = np.zeros(folds)
    val_acc = np.zeros(folds)
    val_loss = np.zeros(folds)

    for epoch in range(epochs):
        splits = [1/folds] * folds
        tv_folds = torch.utils.data.random_split(tv_data, splits)
        print("Epoch: ",epoch)
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
                outc_idx = outcome_to_index[outc]
                outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
                #outcomes = BackPain_Convert(outcomes)
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
                cur_cor = calc_cor(outputs, outcomes)
                #print("      cur_cor = ", cur_cor)
                #print("    --- End ---")
                fold_trn_acc += cur_cor
                fold_trn_loss += loss.item()
                fold_data += len(outcomes)
            trn_acc[fold] = float(fold_trn_acc) / fold_data
            trn_loss[fold] = float(fold_trn_loss) / (i+1)
            val_acc[fold], val_loss[fold] = validate(net, val_load, criterion)
            print(("Fold {}: Train Acc: {}, Train Loss: {} | "+"Val Acc: {}, Val Loss: {}").format(fold + 1, trn_acc[fold], trn_loss[fold], val_acc[fold], val_loss[fold]))

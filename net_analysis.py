#Net Analysis
#Functions to analyze/evaluate trained nets

import torch
from net_trainer import *
from data_loader import * #temp for debugging
from test_nets import * #temp for debugging
import random #temp for debugging

index_to_predictor = {
        0: "Age",
        1: "BMI",
        2: "PHQ9Score",
        3: "Ncomorbidities",
        4: "ODITotalScore_Baseline",
        5: "BackPain_Baseline",
        6: "LegPain_Baseline",
        7: "Sex",
        8: "SympDurat",
        9: "Married",
        10: "Education",
        11: "Smoke",
        12: "Exercise",
        13: "WorkStatus",
        14: "Chiro_New",
        15: "Physio_New",
        16: "Trainer",
        17: "PainMed",
        18: "Inflammatory",
        19: "MuscleRelax",
        20: "ECBackPain",
        21: "ECLegPain",
        22: "ECIndependence",
        23: "ECSportAct",
        24: "ECPhysCapac",
        25: "ECSocial",
        26: "ECWellBeing",
        27: "expbackpain",
        28: "explegpain",
        29: "expindependence",
        30: "expsports",
        31: "expphyscap",
        32: "expsocial",
        33: "expwellbeing",
        34: "chiro",
        35: "physio",
        }

predictor_to_range = {
        "Age": 100,
        "BMI": 60,
        "PHQ9Score": 27,
        "Ncomorbidities": 24,
        "ODITotalScore_Baseline": 100,
        "BackPain_Baseline": 10,
        "LegPain_Baseline": 10,
        "Sex": 1,
        "SympDurat": 1,
        "Married": 1,
        "Education": 1,
        "Smoke": 1,
        "Exercise": 3,
        "WorkStatus": 3,
        "Chiro_New": 3,
        "Physio_New": 3,
        "Trainer": 3,
        "PainMed": 2,
        "Inflammatory": 2,
        "MuscleRelax": 2,
        "ECBackPain": 3,
        "ECLegPain": 3,
        "ECIndependence": 3,
        "ECSportAct": 3,
        "ECPhysCapac": 3,
        "ECSocial": 3,
        "ECWellBeing": 3,
        "expbackpain": 1,
        "explegpain": 1,
        "expindependence": 1,
        "expsports": 1,
        "expphyscap": 1,
        "expsocial": 1,
        "expwellbeing": 1,
        "chiro": 1,
        "physio": 1,
        }


tvs_data = load_data()
tvs_load = DataLoader(tvs_data, batch_size=len(tvs_data))
for i, data in enumerate(tvs_load):
    inputs = data['predictors']
    outcomes = data['outcomes']
    index_to_std = {
        0: inputs[:,0].std().item(),
        1: inputs[:,1].std().item(),
        2: inputs[:,2].std().item(),
        3: inputs[:,3].std().item(),
        4: inputs[:,4].std().item(),
        5: inputs[:,5].std().item(),
        6: inputs[:,6].std().item(),
        7: inputs[:,7].std().item(),
        8: inputs[:,8].std().item(),
        9: inputs[:,9].std().item(),
        10: inputs[:,10].std().item(),
        11: inputs[:,11].std().item(),
        12: inputs[:,12].std().item(),
        13: inputs[:,13].std().item(),
        14: inputs[:,14].std().item(),
        15: inputs[:,15].std().item(),
        16: inputs[:,16].std().item(),
        17: inputs[:,17].std().item(),
        18: inputs[:,18].std().item(),
        19: inputs[:,19].std().item(),
        20: inputs[:,20].std().item(),
        21: inputs[:,21].std().item(),
        22: inputs[:,22].std().item(),
        23: inputs[:,23].std().item(),
        24: inputs[:,24].std().item(),
        25: inputs[:,25].std().item(),
        26: inputs[:,26].std().item(),
        27: inputs[:,27].std().item(),
        28: inputs[:,28].std().item(),
        29: inputs[:,29].std().item(),
        30: inputs[:,30].std().item(),
        31: inputs[:,31].std().item(),
        32: inputs[:,32].std().item(),
        33: inputs[:,33].std().item(),
        34: inputs[:,34].std().item(),
        35: inputs[:,35].std().item(),
        }

def calc_in_error(true, noise, p):
    total_err = 0
    #print("true in = ", true)
    #print("noise in = ", noise)
    if len(true) == len(noise):
        for i in range(0,len(true)):
            outn = noise[i].item()
            outt = true[i].item()
            #print("int[",i,"] = ", outt)
            #print("inn[",i,"] = ", outn)
            #print("err[",i,"] = ", abs(outt - outn)/outcome_to_range[outc])

            """
            if p == 0 or p == 1:
                #print("err[",i,"] = ", abs(outt - outn) / outt)
                total_err+= abs(outt - outn) / outt
            else:
                #print("err[",i,"] = ", abs(outt - outn) / predictor_to_range[index_to_predictor[p]])
                total_err+= abs(outt - outn) / predictor_to_range[index_to_predictor[p]]
            """
            total_err+= abs(outt - outn) / predictor_to_range[index_to_predictor[p]]

            
            #print("-")
            
    else:
        print("ERROR: calc_err(): outputs and outcomes different size")
        return 0.0
    #print("total_err = ", total_err)
    return total_err

def calc_error(true, noise, outc):
    total_err = 0
    if len(true) == len(noise):
        for i in range(0,len(true)):
            outn = noise[i].item()
            outt = true[i].item()
            #print("outt[",i,"] = ", outt)
            #print("outn[",i,"] = ", outn)
            #print("err[",i,"] = ", abs(outt - outn)/outcome_to_range[outc])
            #print("-")

            total_err+= abs(outt - outn) / outcome_to_range[outc]
            
    else:
        print("ERROR: calc_err(): outputs and outcomes different size")
        return 0.0
    #print("total_err = ", total_err)
    return total_err

def evaluate_net(net, dataset, outc, save_num, one_point):
    print("--- Starting Evaluation ---")
    path = "../"+outc+"Net"+str(save_num)
    print("path = ", path)

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

        num_pred = len(inputs[0])
        #print(" Inputs.size = ", inputs.size())
        #print(" Inputs.size[0] = ", num_pred)
        #print(" Inputs.size[1] = ", len(inputs))
        #print(" Inputs = ", inputs)

        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        true_outputs = net(inputs)
        print(" true_outputs = ", true_outputs)

        noise_out_err = np.zeros(num_pred)
        noise_in_err = np.zeros(num_pred)

        cont = 0
        nomi = 0
        ordi = 0
        for p in range(num_pred): # Loop through all predictors
            ninputs = inputs.clone().detach()
            if p < 7: # Continuous predictors
                #continue
                cont = cont + 1
                print(p)
                print(index_to_predictor[p])
                #print(("{} : {}").format(index_to_predictor[p], inputs[:,p]))
                print("std = ", ninputs[:,p].std().item())
                print("idx std = ", index_to_std[p])
                std = ninputs[:,p].std().item()

                for j in range(len(ninputs)): #Add noise to each data point in batch
                    noise = torch.randn(1)
                    noiset = noise * index_to_std[p]
                    if torch.cuda.is_available():
                        noiset = noiset.cuda()
                    #print("noise = ", noise)
                    #print("noise * std = ", noiseb)
                    #print("noise * tot_std = ", noiset)
                    #print("ninputs[j,p] = ", ninputs[j,p])
                    ninputs[j,p] = ninputs[j,p] + noiset
                    #print("ninputs[j,p] = ", ninputs[j,p])

                noise_in_err[p] = calc_in_error(inputs[:,p], ninputs[:,p], p)
                #print("totinerr = ", noise_in_err[p])
                print("totinerr/d = ", noise_in_err[p]/len(ninputs))

                noise_outputs = net(ninputs)
                #print(" noise_outputs = ", noise_outputs)
                #print(" true_outputs = ", true_outputs)

                noise_out_err[p] = calc_error(true_outputs, noise_outputs, outc)
                #print("toterr = ", noise_out_err[p])
                #print("toterr/d = ", noise_out_err[p]/len(ninputs))


            else: # Categorical Predictors
                if (p >= 7 and p <= 11) or (p >= 27 and p <= 35): # Binary/2Category Predictors
                    nomi = nomi + 1
                    print(p)
                    print(index_to_predictor[p])
                    #print(("{} : {}").format(index_to_predictor[p], inputs[:,p]))
                    print("std = ", ninputs[:,p].std().item())
                    print("idx std = ", index_to_std[p])
                    std = ninputs[:,p].std().item()

                    print("in = ", ninputs[:,p])
                    for l in range(len(ninputs)):
                        if ninputs[l,p] == 0:
                            ninputs[l,p] = 1
                        else:
                            ninputs[l,p] = 0
                    print("in = ", ninputs[:,p])

                    noise_outputs = net(ninputs)
                    #print(" noise_outputs = ", noise_outputs)
                    #print(" true_outputs = ", true_outputs)
                    #exit()
                else:
                    ordi = ordi + 1
                    print(p)
                    print(index_to_predictor[p])
                    #print(("{} : {}").format(index_to_predictor[p], ninputs[:,p]))
                    print("std = ", ninputs[:,p].std().item())
                    print("idx std = ", index_to_std[p])
                    std = ninputs[:,p].std().item()

                    for j in range(len(ninputs)): #Add noise to each data point in batch
                        noise = torch.randn(1)
                        #noiset = noise * index_to_std[p]
                        if torch.cuda.is_available():
                            #noiset = noiset.cuda()
                            noise = noise.cuda()
                        #print("noise = ", noise)
                        #print("noise * std = ", noiseb)
                        #print("noise * tot_std = ", noiset)
                        #print("ninputs[j,p] = ", ninputs[j,p])
                        ninputs[j,p] = ninputs[j,p] + noise
                        #print("ninputs[j,p]n = ", ninputs[j,p])
                        ninputs[j,p] = torch.round(ninputs[j,p])
                        ninputs[j,p] = max(ninputs[j,p], 0)
                        ninputs[j,p] = min(ninputs[j,p], predictor_to_range[index_to_predictor[p]])
                        #print("rninputs[j,p]n = ", ninputs[j,p])

                    noise_in_err[p] = calc_in_error(inputs[:,p], ninputs[:,p], p)
                    #print("totinerr = ", noise_in_err[p])
                    print("totinerr/d = ", noise_in_err[p]/len(ninputs))

                    noise_outputs = net(ninputs)
                    #print(" noise_outputs = ", noise_outputs)
                    #print(" true_outputs = ", true_outputs)

                    noise_out_err[p] = calc_error(true_outputs, noise_outputs, outc)
                    #print("toterr = ", noise_out_err[p])
                    print("toterr/d = ", noise_out_err[p]/len(ninputs))


                    #exit()
        print("cont = ", cont)
        print("nomi = ", nomi)
        print("ordi = ", ordi)

        """
        age = inputs[:,p]
        print("age = ", age)
        age = age * 2
        print("age = ", age)
        print("i:p = ", inputs[:,p])
        """


        if torch.cuda.is_available() == True:
            inputs = inputs.cuda()
            outcomes = outcomes.cuda()
            #print("Using cuda")
        else:
            print("Using CPU")

        outputs = net(inputs)

        print(" Outcomes.size = ", outcomes.size())
        print(" Outcomes = ", outcomes)

        print(" Output.size = ", outputs.size())
        print(" Output = ", outputs)

        outputs2 = net(inputs)

        print(" Output2.size = ", outputs2.size())
        print(" Output2 = ", outputs2)

        if one_point: #Break after first batch if one_point True
            if i==0:
                break

        
#temp for debugging

folds = 10
tvs_data = load_data()
splits = [1/folds] * folds
tvs_folds = torch.utils.data.random_split(tvs_data, splits)
fold = random.randrange(folds)
print("Test Fold: ", fold)

ts_data = tvs_folds[fold]
ts_load = DataLoader(ts_data, batch_size=10, shuffle=True, drop_last=True)

outc = "EQ_IndexTL12"
save_num = 0
evaluate_net(eqidxtl12_net(), ts_load, outc, save_num, True)

#Net Analysis
#Functions to analyze/evaluate trained nets

import torch
from net_trainer import *
from data_loader import *
#from test_nets import * #temp for debugging
#import random #temp for debugging

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
        "Exercise": 2,
        "WorkStatus": 2,
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

def bi_Convert(outcomes):
    ret = outcomes.clone().detach()
    for i in range(len(outcomes)):
        if outcomes[i] == 1:
            ret[i] = 0
        if outcomes[i] == 2:
            ret[i] = 1
    return ret

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

def calc_error(true, noise, outc, outcomes):
    outc_idx = outcome_to_index[outc]
    total_err = 0
    if len(true) == len(noise):
        for i in range(0,len(true)):
            if outc_idx == 3 or outc_idx == 5:
                outid = int(outcomes[i].item())
                #print("outid = ", outid)
                outn = noise[i,outid].item()
                outt = true[i,outid].item()
            else:
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

def calc_corerror(true, noise, outc, outcomes):
    outc_idx = outcome_to_index[outc]
    total_err = 0
    if len(true) == len(noise):
        for i in range(0,len(true)):
            temp_err = 0.0
            for j in range(0,len(true[i])):
            
                outn = noise[i,j].item()
                outt = true[i,j].item()
            #print("outt[",i,"] = ", outt)
            #print("outn[",i,"] = ", outn)
            #print("err[",i,"] = ", abs(outt - outn)/outcome_to_range[outc])
            #print("-")

                temp_err+= abs(outt - outn) / outcome_to_range[index_to_outcome[j]]
            total_err+= temp_err/6
            
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

    for dt in dataset:
        #print("len data = ", len(dataset))
        #print("len pred = ", len(dt['predictors'].clone().detach()[1]))
        lp = len(dt['predictors'].clone().detach()[1])
        break
    
    noise_in_errl = np.zeros((len(dataset),lp))
    noise_out_errl = np.zeros((len(dataset),lp))
    net_true_errl = np.zeros(len(dataset))

    #Either add break or counter to limit from single data point to entire test set used and averaged to find the critical predictors
    for i, data in enumerate(dataset):
        inputs = data['predictors']
        outcomes = data['outcomes']
        if outc_idx != 6:
            outcomes = outcomes[:,outc_idx].unsqueeze(1).to(torch.float32)
            if (outc_idx == 3):
                outcomes = bi_Convert(outcomes)
        else:
            outcomes = outcomes.to(torch.float32)

        #print(" outcomes = ", outcomes)

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
        #print(" true_outputs = ", true_outputs)
        for q in range(100):
            temp_outputs = net(inputs)
            if outc_idx == 6:
                net_true_errl[i] += calc_corerror(true_outputs, temp_outputs, outc, outcomes) / len(inputs)
            else:
                net_true_errl[i] += calc_error(true_outputs, temp_outputs, outc, outcomes) / len(inputs)
        net_true_errl[i] = net_true_errl[i] / 100

        #print("net_true_errl[i] = ", net_true_errl[i])


        for p in range(num_pred): # Loop through all predictors
            ninputs = inputs.clone().detach()
            if p < 7: # Continuous predictors
                #print(p)
                #print(index_to_predictor[p])
                #print(("{} : {}").format(index_to_predictor[p], inputs[:,p]))
                #print("std = ", ninputs[:,p].std().item())
                #print("idx std = ", index_to_std[p])
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

                noise_in_errl[i,p] = calc_in_error(inputs[:,p], ninputs[:,p], p) / len(ninputs)
                #print("totinerr = ", noise_in_errl[i,p])

                noise_outputs = net(ninputs)
                #print(" noise_outputs = ", noise_outputs)
                #print(" true_outputs = ", true_outputs)
                if outc_idx == 6:
                    noise_out_errl[i,p] = calc_corerror(true_outputs, noise_outputs, outc, outcomes) / len(ninputs)
                else:
                    noise_out_errl[i,p] = calc_error(true_outputs, noise_outputs, outc, outcomes) / len(ninputs)
                #print("toterr = ", noise_out_errl[i,p])


            else: # Categorical Predictors
                if (p >= 7 and p <= 11) or (p >= 27 and p <= 35): # Binary/2Category Predictors
                    #print(p)
                    #print(index_to_predictor[p])
                    #print(("{} : {}").format(index_to_predictor[p], inputs[:,p]))
                    #print("std = ", ninputs[:,p].std().item())
                    #print("idx std = ", index_to_std[p])
                    std = ninputs[:,p].std().item()

                    #print("in = ", ninputs[:,p])
                    for l in range(len(ninputs)):
                        if ninputs[l,p] == 0:
                            ninputs[l,p] = 1
                        else:
                            ninputs[l,p] = 0
                    #print("in = ", ninputs[:,p])

                    noise_outputs = net(ninputs)
                    #print(" noise_outputs = ", noise_outputs)
                    #print(" true_outputs = ", true_outputs)

                    noise_in_errl[i,p] = 0.0 #calc_in_error(inputs[:,p], ninputs[:,p], p) / len(ninputs)
                    #print("totinerr = ", noise_in_errl[i,p])
                    if outc_idx == 6:
                        noise_out_errl[i,p] = calc_corerror(true_outputs, noise_outputs, outc, outcomes) / len(ninputs)
                    else:
                        noise_out_errl[i,p] = calc_error(true_outputs, noise_outputs, outc, outcomes) / len(ninputs)
                    #print("toterr = ", noise_out_errl[i,p])

                    #exit()
                else:
                    #print(p)
                    #print(index_to_predictor[p])
                    #print(("{} : {}").format(index_to_predictor[p], ninputs[:,p]))
                    #print("std = ", ninputs[:,p].std().item())
                    #print("idx std = ", index_to_std[p])
                    std = ninputs[:,p].std().item()

                    for k in range(len(ninputs)): #Add noise to each data point in batch
                        noise = torch.randn(1)
                        #noiset = noise * index_to_std[p]
                        if torch.cuda.is_available():
                            #noiset = noiset.cuda()
                            noise = noise.cuda()
                        #print("noise = ", noise)
                        #print("noise * std = ", noiseb)
                        #print("noise * tot_std = ", noiset)
                        #print("ninputs[k,p] = ", ninputs[k,p])
                        ninputs[k,p] = ninputs[k,p] + noise
                        #print("ninputs[k,p]n = ", ninputs[k,p])
                        ninputs[k,p] = torch.round(ninputs[k,p])
                        ninputs[k,p] = max(ninputs[k,p], 0)
                        ninputs[k,p] = min(ninputs[k,p], predictor_to_range[index_to_predictor[p]])
                        #print("rninputs[k,p]n = ", ninputs[k,p])

                    noise_in_errl[i,p] = calc_in_error(inputs[:,p], ninputs[:,p], p) / len(ninputs)
                    #print("totinerr = ", noise_in_errl[i,p])

                    noise_outputs = net(ninputs)
                    #print(" noise_outputs = ", noise_outputs)
                    #print(" true_outputs = ", true_outputs)
                    if outc_idx == 6:
                        noise_out_errl[i,p] = calc_corerror(true_outputs, noise_outputs, outc, outcomes) / len(ninputs)
                    else:
                        noise_out_errl[i,p] = calc_error(true_outputs, noise_outputs, outc, outcomes) / len(ninputs)
                    #print("toterr = ", noise_out_errl[i,p])


                    #exit()

        if one_point: #Break after first batch if one_point True
            if i==0:
                break
    
    #print("net_true_errl = ", net_true_errl)
    #print("noise_in_errl = ", noise_in_errl)
    #print("noise_out_errl = ", noise_out_errl)

    if one_point:
        avg_net_true_err = net_true_errl[0]
        avg_noise_in_err = noise_in_errl[0]
        avg_noise_out_err = noise_out_errl[0]
    else:
        avg_net_true_err = np.mean(net_true_errl)
        avg_noise_in_err = np.zeros(lp)
        avg_noise_out_err = np.zeros(lp)
        for r in range(lp):
            avg_noise_in_err[r] = np.mean(noise_in_errl[:,r])
            avg_noise_out_err[r] = np.mean(noise_out_errl[:,r])

    #print(" avg_net_true_err = ", avg_net_true_err)
    #print("avg_noise_in_err = ", avg_noise_in_err)
    #print(" avg_noise_out_err = ", avg_noise_out_err)

    scaled_avg_noise_out_err = avg_noise_out_err / avg_net_true_err
    #print(" scaled_avg_noise_out_err = ", scaled_avg_noise_out_err)

    sub_scaled_avg_noise_out_err = np.abs(scaled_avg_noise_out_err - 1)
    #print(" sub_scaled_avg_noise_out_err = ", sub_scaled_avg_noise_out_err)

    vimp = sub_scaled_avg_noise_out_err * 10
    print(" vimp = ", vimp)

    for f in range(lp):
        if vimp[f] >= 1:
            print((" Critical Predictor: {} | "+"VI Score: {:.5f}").format(index_to_predictor[f], vimp[f]))

    print("--- Finished Evaluation ---")
    return vimp

        
#temp for debugging
"""
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
evaluate_net(eqidxtl12_net(), ts_load, outc, save_num, False)

outc = "ODI4_Final"
save_num = 0
evaluate_net(recovery_net(), ts_load, outc, save_num, False)

outc = "ODIScore"
save_num = 0
evaluate_net(odiscore_net(), ts_load, outc, save_num, False)
"""
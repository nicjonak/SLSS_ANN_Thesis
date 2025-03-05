#Data Loader
#Open/Read CSORN Excel and load into pytorch dataset/dataloader

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#Custom Dataset class for the CSORN data
class SLSSDataset(Dataset):
    def __init__(self, xls_file, transform=None):
        self.data_tensor = torch.from_numpy(pd.read_excel(xls_file).to_numpy())
    
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        pred = self.data_tensor[idx,1:-11]
        outc = self.data_tensor[idx,-11:-5]
        sample = {'predictors': pred, 'outcomes': outc}
        return sample

#load_data function, inputs batch size and test split percentage, outputs randomly split train/val and test datasets
def load_data(ts_per):
    xls_file = '../CSORN_Edit.xls'
    #df = pd.read_excel(xls_file)
    #print("df[0] = ", df.iloc[0])
    data_set = SLSSDataset(xls_file)
    #print("Dataset[0] = ", data_set[0])

    splits = len(data_set)
    ts_split = int(splits * ts_per)
    tv_split = splits - ts_split

    print("Total Data: ",splits)
    print("Train/Val Data: ",tv_split)
    print("Test Data: ",ts_split)

    if (ts_per == 0 or ts_per == 0.0):
        """
        tv_load = DataLoader(data_set, batch_size=batch, shuffle=True)
        return tv_load
        """
        return data_set
    else:
        tv_data, ts_data = torch.utils.data.random_split(data_set, [tv_split, ts_split])
        """
        ret = torch.utils.data.random_split(data_set, [tv_split, ts_split])
        print("ret = ", ret)
        print("ret[0] = ", ret[0])
        comb = torch.utils.data.ConcatDataset(ret)
        print("comb = ", comb)
        """
        """
        tv_load = DataLoader(tv_data, batch_size=batch, shuffle=True)
        ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True)
        return tv_load, ts_load
        """
        return tv_data, ts_data

"""
#batchs = 10
tsper = 0.01
ret = load_data(tsper)
print("ret = ", ret)

#SLSS_tv_data, SLSS_ts_data = load_data(tsper)
#print("Batch Size = ", batchs)
print("ts_per = ", tsper)

for i, sample in enumerate(ret[1]):
    print("Test Sample: ", i)
    print("pred = ",sample['predictors'],", outc = ",sample['outcomes'])
    print("pred.size = ", sample['predictors'].size(), ", outc.size = ", sample['outcomes'].size())
    print("len(outcomes) = ", len(sample['outcomes']))
    break
"""

#Data Loader
#Open/Read CSORN Excel and load into pytorch dataset/dataloader

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SLSSDataset(Dataset):
    def __init__(self, xls_file, transform=None):
        self.data_tensor = torch.from_numpy(pd.read_excel(xls_file).to_numpy())
    
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        pred = self.data_tensor[idx,:-11]
        outc = self.data_tensor[idx,-11:-5]
        sample = {'predictors': pred, 'outcomes': outc}
        return sample

def load_data(batch, ts_per):
    xls_file = '../CSORN_Edit.xls'
    data_set = SLSSDataset(xls_file)

    splits = len(data_set)
    ts_split = int(splits * ts_per)
    tv_split = splits - ts_split

    print("Total Data: ",splits)
    print("Train/Val Data: ",tv_split)
    print("Test Data: ",ts_split)

    if (ts_per == 0):
        tv_load = DataLoader(data_set, batch_size=batch, shuffle=True)
        return tv_load
    else:
        tv_data, ts_data = torch.utils.data.random_split(data_set, [tv_split, ts_split])
        tv_load = DataLoader(tv_data, batch_size=batch, shuffle=True)
        ts_load = DataLoader(ts_data, batch_size=batch, shuffle=True)
        return tv_load, ts_load
"""
SLSS_tv_load, SLSS_ts_load = load_data(10, 0.01)


for i, sample in enumerate(SLSS_ts_load):
    print("Test Sample ",i,": pred = ",sample['predictors'],", outc = ",sample['outcomes'])
"""

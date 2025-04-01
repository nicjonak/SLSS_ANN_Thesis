#Data Loader
#Open/Read CSORN Excel and load into pytorch dataset/dataloader

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#Custom Dataset class for the CSORN data
class SLSSDataset(Dataset):
    def __init__(self, xls_file, transform=None):
        self.data_tensor = torch.from_numpy(pd.read_excel(xls_file).to_numpy()) #Read CSORN excel file into numpy array using pandas
    
    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        pred = self.data_tensor[idx,1:-11] #Separate predictors from data
        outc = self.data_tensor[idx,-11:-5] #Separate outcomes from data
        sample = {'predictors': pred, 'outcomes': outc}
        return sample


def load_data():
    xls_file = '../CSORN_Edit.xls' #Specify path to CSORN excel file
    """
    #df = pd.read_excel(xls_file)
    #print("df[0] = ", df.iloc[0])
    """
    data_set = SLSSDataset(xls_file) #Read CSORN data into dataset
    """
    #print("Dataset[0] = ", data_set[0])
    """
    #Specify data splits
    splits = len(data_set)
    print("Total Data: ",splits)
    

    
    return data_set

#Below is debuggig/test/etc code

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

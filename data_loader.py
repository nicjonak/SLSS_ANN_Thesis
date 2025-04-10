#Data Loader
#Open/Read CSORN Excel File and load into pytorch dataset/dataloader

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
        sample = {'predictors': pred, 'outcomes': outc} #Data structured as tensors within a dictionary

        return sample

#Function to load data from XLS file into torch dataset
def load_data():
    xls_file = '../CSORN_Edit.xls' #Specify path to CSORN excel file (edit with your path)
    data_set = SLSSDataset(xls_file) #Read CSORN data into dataset
    splits = len(data_set)
    print("Total Data: ",splits) #Print total size of dataset for curiosity/sanity (may be commented out/removed)

    return data_set

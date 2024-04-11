import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.special import softmax


class HeatMapDataset(Dataset):
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        heatmap_str = self.df.iloc[idx, 10][1:-1]

        data = np.fromstring(heatmap_str, sep=',') 
        
        height = 56
        width = 80
        channels = 3
        
        #-------------------------------------------------

        reshape_old = np.array(data).reshape(channels, height, width)
        
        reshape_old[0] = softmax(reshape_old[0])
        reshape_old[1] = softmax(reshape_old[1])
        reshape_old[2] = softmax(reshape_old[2])
        
        #-------------------------------------------------
        
        reshape_new = reshape_old.transpose((1,2,0))

        output = reshape_new.copy()
        
        r =  reshape_new[:,:,0]
        g =  reshape_new[:,:,1]
        b = reshape_new[:,:,2]

        r_min = r.min()
        g_min = g.min()
        b_min = b.min()

        r_max = r.max()
        g_max = g.max()
        b_max = b.max()

        output[:,:,0] = (reshape_new[:,:,2]-b_min)/(b_max-b_min)
        output[:,:,1] = (reshape_new[:,:,1]-g_min)/(g_max-g_min)
        output[:,:,2] = (reshape_new[:,:,0]-r_min)/(r_max-r_min)

        output = np.where(output > 0.1, output, np.zeros_like(output))
        
        reshape = output.transpose(2, 0, 1)
        
        
        targets = self.df.iloc[idx, 6:8].values.astype(np.float32)
        
        
        
        return torch.tensor(reshape, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
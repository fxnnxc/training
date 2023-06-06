import torch 
from torch.utils.data import Dataset 
import pickle 
import os 
import numpy as np 
target = 'split'


DATA_STATS  = { 
               'mean' : 2.4092,
               'std' : 13.8919
               }
DYNAMCIS_STATS  = { 
               'mean' : -1.1933e-09,
               'std' : 1.000
               }

class AKIDataset(Dataset): 
    def __init__(self, data, is_train):
        self.is_train = is_train
        target_cols1 = ['d1_1_aki', 'd1_2_aki', 'd1_3_aki']
        target_cols2 = ['d2_1_aki', 'd2_2_aki', 'd2_3_aki']
        dynamic = []
        for c in data.columns:
            for d in range(1, 8):
                for t in range(1, 4):
                    if f'd{d}_{t}_' in c and 'aki' not in c:
                        dynamic.append(c)
                        
        target_cols = target_cols1 + target_cols2 + ['split', 'split1', 'split2']

        
        self.x_data = torch.Tensor(data.loc[:, data.columns.difference(target_cols).difference(dynamic)].values)
        self.x_dy = torch.Tensor(data.loc[:, dynamic].values)
        self.y_data = torch.Tensor(data.loc[:, target].values)
        self.y_dy = torch.Tensor(data.loc[:, 'd2_3_creatinine'].values)

        # --- Normalize --- 
        self.x_data =  (self.x_data- DATA_STATS['mean'])/DATA_STATS['std']
        self.x_dy =  (self.x_dy- DYNAMCIS_STATS['mean'])/DYNAMCIS_STATS['std']
        

    def __len__(self): 
        return len(self.x_data)
    
    def __getitem__(self, idx): 
        # return {'data': self.x_data[idx], 'dynamic': self.x_dy[idx], 
        #         'labels': self.y_data[idx], 'ground': self.y_dy[idx]}
        
        x1 = self.x_data[idx]
        x2 = self.x_dy[idx]
        max_len = max(x1.size(0), x2.size(0))
        zeros = torch.zeros(2, max_len)      
        zeros[0, -x1.size(0):] = x1.data.clone() 
        zeros[1, :x2.size(0)]  = x2.data.clone() 
        # if self.is_train:
        #         # zeros = torch.flip(zeros, dims=(1,))
        #         zeros[0, np.random.randint(max_len)] = 0.0 # drop out 
        return zeros, int(self.y_data[idx].item())

def get_jihyeon_dataset(root, resize=None, transforms=None):
    train = AKIDataset(pickle.load(open(os.path.join(root, 'train.pickle') ,'rb')), is_train=True)
    valid = AKIDataset(pickle.load(open(os.path.join(root, 'valid.pickle') ,'rb')), is_train=False)
    test  = AKIDataset(pickle.load(open(os.path.join(root, 'test.pickle') ,'rb')), is_train=False)
    
    return train, valid, {}


if __name__ == "__main__":
    train, valid, info =  get_jihyeon_dataset("/data4/bumjin/jihyeon_data")
    print(len(train))
    print(len(valid))
    print(train[0]['data'].size())
    print(train[0]['labels'])

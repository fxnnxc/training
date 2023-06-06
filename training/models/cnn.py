import torch 
import torch.nn as nn 
import numpy as np 

def layer_init(layer, std=2**(1/2), bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

    
class JihyeonCNNClassifier(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        
        self.embedding_1 = nn.Sequential(
            nn.Dropout(0.2),
            layer_init(nn.Linear(1, 32,)),
            nn.ReLU(),
            layer_init(nn.Linear(32, hidden_dim,)),
            nn.ReLU(),
        )
        self.embedding_2 = nn.Sequential(
            nn.Dropout(0.2),
            layer_init(nn.Linear(1, 32,)),
            nn.ReLU(),
            layer_init(nn.Linear(32, hidden_dim,)),
            nn.ReLU(),
        )
        
        
        self.shared_encoder = nn.Sequential(
            nn.BatchNorm1d(hidden_dim*2),
            layer_init(nn.Conv1d(hidden_dim*2, 256, 3, 1, 1)),
            nn.ReLU(),
            nn.MaxPool1d(3,),
            nn.BatchNorm1d(256),
            layer_init(nn.Conv1d(256, 512, 3, 1,1)),
            nn.ReLU(),
            nn.MaxPool1d(3,),
            nn.BatchNorm1d(512),
            layer_init(nn.Conv1d(512, 1024, 3, 1,1)),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
                        layer_init(nn.Linear(1024, 128,)),
                        nn.ReLU(),
                        layer_init(nn.Linear(128, 64,)),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        layer_init(nn.Linear(64, 2))
                        )
        
    def forward(self, x):
        # x = self.embedding(x)  # B T D

        x = x.permute(0, 2, 1) # B T 2
        x_1 = x[:,:,0].unsqueeze(-1)
        x_1 = self.embedding_1(x_1)
        x_2 = x[:,:,1].unsqueeze(-1)
        x_2 = self.embedding_2(x_2)
        x = torch.cat([x_1, x_2], dim=-1)
        x = x.permute(0, 2, 1) # B D T 
        x = self.shared_encoder(x)
        x = x.mean(dim=-1)
        x = self.fc(x) 
        return x 
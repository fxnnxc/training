import torch 
import torch.nn as nn 
import numpy as np 

def layer_init(layer, std=2**(1/2), bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class JihyeonLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(2, hidden_dim,)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim,)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim,)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim,)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim,)),
            nn.ReLU(),
        )
        
        # self.shared_encoder = nn.Sequential(
        #     nn.BatchNorm1d(hidden_dim),
        #     layer_init(nn.Conv1d(hidden_dim, 128, 3, 1, 1)),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     layer_init(nn.Conv1d(128, 256, 3, 1,1)),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     layer_init(nn.Conv1d(256, hidden_dim, 3, 1,1)),
        #     nn.ReLU(),
        # )
        
        self.num_rnn_layers = 4
        self.rnn_hidden_dim = 128
        self.rnn = nn.LSTM(hidden_dim, self.rnn_hidden_dim, self.num_rnn_layers, batch_first=True)
        
        self.fc = nn.Sequential(
                        layer_init(nn.Linear(self.rnn_hidden_dim, hidden_dim,)),
                        nn.ReLU(),
                        layer_init(nn.Linear(hidden_dim, hidden_dim,)),
                        nn.ReLU(),
                        layer_init(nn.Linear(hidden_dim, hidden_dim,)),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 2))
        
    def forward(self, x):
        # x = torch.cat([x1, x2], dim=1)
        x = x.permute(0, 2, 1) # B T 2
        x = self.embedding(x)  # B T D 
        h0 = torch.zeros(self.num_rnn_layers, x.size(0), self.rnn_hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_rnn_layers, x.size(0), self.rnn_hidden_dim, device=x.device)
        output, hn = self.rnn(x, (h0, c0))
        x = output[:,-1]
        x = self.fc(x) 
        return x 
    
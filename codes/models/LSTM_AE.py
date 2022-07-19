import torch
import torch.nn as nn
import numpy as np
import pandas as pd

torch.manual_seed(0)

class _Encoder(nn.Module):
    def __init__(self, no_features, embedding_size):
        super().__init__()
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.rnn1 = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
        )
            
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        x, (_, _) = self.rnn1(x.view(1, 1, -1))
        x, (hidden_n, _) = self.rnn2(x.view(1, -1))
        return hidden_n
    
    
class _Decoder(nn.Module):
    def __init__(self, no_features, input_dim):
        super().__init__()
        self.no_features = no_features
        self.hidden_size = (2 * input_dim)
        self.input_dim = input_dim
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim ,
            num_layers=2,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_size, no_features)
        
    def forward(self, x):
        x, (_, _) = self.rnn1(x.view(1, 1, -1))
        x, (_, _) = self.rnn2(x)
        x = x.reshape((self.hidden_size))
        return self.output_layer(x)
    

class LSTM_AE(nn.Module):
    def __init__(self, feats, config):
        super(LSTM_AE, self).__init__()
        self.name = 'LSTM_AE'
        self.lr = config['learning_rate']
        self.no_features = feats
        self.embedding_dim = config['embedding_dim']
        self.encoder = _Encoder(self.no_features, self.embedding_dim)
        self.decoder = _Decoder(self.no_features, self.embedding_dim)
        
    def forward(self, x):
        torch.manual_seed(0)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
        
        
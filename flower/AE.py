import torch.nn as nn

class AE(nn.Module):
    def __init__(self, feats, config):
        super(AE,self).__init__()
        self.name = 'AE'
        self.lr = config['learning_rate']
        self.n_window = config['num_window']
        n = feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(n, int(0.75*n)),
            nn.Tanh(),
            nn.Linear(int(0.75*n), int(0.5*n)),
            nn.Tanh(),
            nn.Linear(int(0.5*n), int(0.33*n)),
            nn.Tanh(),
            nn.Linear(int(0.33*n), int(0.25*n)),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(0.25*n), int(0.33*n)),
            nn.Tanh(),
            nn.Linear(int(0.33*n), int(0.5*n)),
            nn.Tanh(),
            nn.Linear(int(0.5*n), int(0.75*n)),
            nn.Tanh(),
            nn.Linear(int(0.75*n), int(n)),
            nn.Tanh(),
        )

        
    def forward(self, x):
        encode = self.encoder(x)
        decoder = self.decoder(encode)
        return decoder
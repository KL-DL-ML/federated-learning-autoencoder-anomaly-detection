import torch 
import torch.nn as nn

class LSTM_Univariate(nn.Module):
	def __init__(self, feats, config):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.001
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)
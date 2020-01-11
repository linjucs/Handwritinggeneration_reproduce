import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as numpy
import math
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli

class UnconditionalModel(nn.Module):
	def __init__(self, input_dim=3, num_layers=3, hidden_dim=400, num_mix_gaussian=20, dropout=0.2, use_cuda=False):
		super(UnconditionalModel, self).__init__()
		#Here we follow alex paper using three LSTM layers
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.n_gaussians = num_mix_gaussian
		self.use_cuda = use_cuda
		self.num_layers = num_layers
		self.dropout = nn.Dropout(p=dropout)
		#first layer
		self.input_size1 = self.input_dim
		self.lstm1 = nn.LSTMCell(input_size= self.input_size1 , hidden_size = self.hidden_dim)

		#second layer
		#Note that here we add skip connections
		self.input_size2 = self.input_dim + self.hidden_dim
		self.lstm2 = nn.LSTMCell(input_size= self.input_size2 , hidden_size = self.hidden_dim)

		#third layer
		#Note that here we add skip connections
		self.input_size3 = self.input_dim + self.hidden_dim
		self.lstm3 = nn.LSTMCell(input_size= self.input_size3 , hidden_size = self.hidden_dim)

		#mixture density outputs
		self.e = nn.Linear(self.hidden_dim*self.num_layers, 1) # end of stroke probability
		self.pi = nn.Linear(self.hidden_dim*self.num_layers, self.n_gaussians) # mixture weights
		self.mu1 = nn.Linear(self.hidden_dim*self.num_layers, self.n_gaussians) #means - 1
		self.sig1 = nn.Linear(self.hidden_dim*self.num_layers, self.n_gaussians) #standard deviations - 1
		self.mu2 = nn.Linear(self.hidden_dim*self.num_layers, self.n_gaussians) #means -2
		self.sig2 = nn.Linear(self.hidden_dim*self.num_layers, self.n_gaussians) #standard deviations -2
		self.ro = nn.Linear(self.hidden_dim*self.num_layers, self.n_gaussians) # correlations

	def forward(self, x, prev_state):
		if self.use_cuda:
			x = x.cuda()
		#drop_in = self.dropout(x)
		x = x.permute(1,0,2)
		seq_len = x.shape[1]
		
		if prev_state is not None:
			h1,c1,h2,c2,h3,c3 = prev_state
		else:
			state = self.init_hidden(x.shape[0])
			h1,c1 = state[0]
			h2,c2 = state[1]
			h3,c3 = state[2]
		lstm1_out = torch.zeros(seq_len, x.shape[0], self.hidden_dim).cuda()
		lstm2_out = torch.zeros(seq_len, x.shape[0], self.hidden_dim).cuda()
		lstm3_out = torch.zeros(seq_len, x.shape[0], self.hidden_dim).cuda()
		for i in range(seq_len):
			h1, c1 = self.lstm1(x[:, i], (h1, c1))
			h2, c2 = self.lstm2(torch.cat((x[:, i], h1), 1), (h2, c2))
			h3, c3 = self.lstm3(torch.cat((x[:, i], h2), 1), (h3, c3))
			lstm1_out[i] = h3
			lstm2_out[i] = h3
			lstm3_out[i] = h3
		lstm1_out = lstm1_out.permute(1,0,2)
		lstm2_out = lstm2_out.permute(1,0,2)
		lstm3_out = lstm3_out.permute(1,0,2)
		prev_state = h1,c1,h2,c2,h3,c3
		hidden_out = torch.cat((lstm1_out, lstm2_out, lstm2_out), 2)

		e = self.e(hidden_out)
		e = 1 / (1 + torch.exp(e))
		pi = self.pi(hidden_out)
		pi = torch.softmax(pi, -1)

		mu1 = self.mu1(hidden_out)
		mu2 = self.mu2(hidden_out)
		sig1 = self.sig1(hidden_out)
		sig1 = torch.exp(sig1)
		sig2 = self.sig2(hidden_out)
		sig2 = torch.exp(sig2)
		ro = self.ro(hidden_out)
		ro = torch.tanh(ro)
		
		return e, pi, mu1, mu2, sig1, sig2, ro, prev_state

	def init_hidden(self, batch_size):
		state = []
		for i in range(self.num_layers):
			if self.use_cuda:
				hidden = (torch.zeros(batch_size, self.hidden_dim).cuda(), 
					torch.zeros(batch_size, self.hidden_dim).cuda())
			else:
				hidden = (torch.zeros(batch_size, self.hidden_dim), 
					torch.zeros(batch_size, self.hidden_dim))
			state.append(hidden)
		return state

	def prediction_loss(self, e, pi, mu1, mu2, sig1, sig2, ro, target, target_mask):
		#first we need to calculate the probability of p(x_t+1|y_t)
		# as we used #mixture gaussian, then we need to copy x1 or x2 to that dim
		target = target.permute(1,0,2)
		target_mask = target_mask.permute(1,0)

		
		x1 = target[:,:,1].repeat(self.n_gaussians,1,1).permute(1, 2, 0) # batch size x seq_len x #gaussians
		x2 = target[:,:,2].repeat(self.n_gaussians, 1,1).permute(1, 2, 0)
		
		x1_norm = ((x1 - mu1) ** 2) / (sig1 * sig1)
		x2_norm = ((x2 - mu2) ** 2) / (sig2 * sig2)
		
		corr_norm = (2*ro*(x1 - mu1)*(x2 - mu2)) / (sig1*sig2)
		z = x1_norm + x2_norm - corr_norm
		
		N = torch.exp(-z/(2*(1-ro**2))) / (2*math.pi*sig1*sig2*((1-ro**2)**.5))

		PR = pi * N
		#PR = self.batched_index_select_mask(PR, 1, target_mask)
		
		eps = 1e-7
		
		e_loss = target[:,:,0] * e.squeeze(-1) + (1-target[:,:,0]) * (1 - e.squeeze(-1))
		
		loss = -torch.log(torch.sum(PR, -1) + eps) - torch.log(e_loss + eps)
		loss = torch.mean(torch.sum(loss, 0))
		return loss

	def generate_samples(self, init_stroke, max_len):
		prev_state = None
		prev_strokes = []
		init_stroke = init_stroke.unsqueeze(0).unsqueeze(0)
		for i in range(max_len):
			
			e, pi, mu1, mu2, sig1, sig2, ro, prev_state = self.forward(init_stroke, prev_state)
		#squezee: 1 x seq_len x dim - > seq_len x dim
			e = e.squeeze(0)
			samples = self.multibivariate_sampling(pi.squeeze(0), mu1.squeeze(0), mu2.squeeze(0), sig1.squeeze(0), sig2.squeeze(0), ro.squeeze(0))
			e = Bernoulli(e)
			e = e.sample()

			#e = e.unsqueeze(-1)
			#print(samples)
			init_stroke = torch.cat((e, samples.cuda()), 1)

			prev_strokes.append(init_stroke)
			init_stroke = init_stroke.unsqueeze(0)
			#print(pre_strokes.shape)
		return torch.stack(prev_strokes, 1)


	def multibivariate_sampling(self, pi, mu1, mu2, sig1, sig2, ro):

		seq_len  = mu1.shape[0]
		 
		random_index = pi.multinomial(1)
		mu1 = mu1.gather(1, random_index)
		mu2 = mu2.gather(1, random_index)
		sig1 = sig1.gather(1, random_index)
		sig2 = sig2.gather(1, random_index)
		samples = torch.zeros(seq_len, 2)
		ro = ro.gather(1, random_index)
	
		for i in range(seq_len):
			mean = torch.tensor([mu1[i], mu2[i]])
			cov = torch.tensor([[sig1[i]**2, ro[i] * sig1[i] * sig2[i]], [ro[i] * sig1[i] * sig2[i], sig2[i]**2]])
			
			m = MultivariateNormal(mean, cov)
			samples[i] = m.sample()
		return samples
	def batched_index_select_mask(self, t, dim, inds):
		dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
		t = t.cuda()
		out = t * dummy# b x e x f
		return out
	

if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	Model = UnconditionalModel(use_cuda=use_cuda).cuda()
	strokes = torch.randn(3, 4, 3).cuda()
	e, pi, mu1, mu2, sig1, sig2, ro = Model(strokes)

	loss = Model.prediction_loss(e, pi, mu1, mu2, sig1, sig2, ro, strokes)
	print(loss.item())
	


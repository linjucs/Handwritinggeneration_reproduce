import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as numpy
import math
import numpy as np
from torch.nn.functional import one_hot
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.bernoulli import Bernoulli
class ConditionalModel(nn.Module):
	def __init__(self, input_dim=3, vocab_size=78, num_layers=3, hidden_dim=400, num_mix_gaussian=20, num_w_gaussian=10, dropout=0.2, bias= 3, use_cuda=False):
		super(ConditionalModel, self).__init__()
		#Here we follow alex paper using three LSTM layers
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.n_gaussians = num_mix_gaussian
		self.w_gaussians = num_w_gaussian
		self.use_cuda = use_cuda
		self.vocab_size = vocab_size
		self.num_layers = num_layers
		self.dropout = nn.Dropout(p=dropout)
		self.bias = bias
		#first layer
		self.input_size1 = self.input_dim + self.vocab_size
		self.lstm1 = nn.LSTMCell(input_size= self.input_size1 , hidden_size = self.hidden_dim)

		# #window layer
		self.w_input = self.hidden_dim
		self.alpha = nn.Linear(self.w_input, self.w_gaussians)
		self.beta = nn.Linear(self.w_input, self.w_gaussians)
		self.offset = nn.Linear(self.w_input, self.w_gaussians)
		#second layer
		#Note that here we add skip connections
		self.input_size2 = self.input_dim + self.hidden_dim + self.vocab_size
		self.lstm2 = nn.LSTMCell(input_size= self.input_size2 , hidden_size = self.hidden_dim)

		#third layer
		#Note that here we add skip connections
		self.input_size3 = self.input_dim + self.hidden_dim + self.vocab_size
		self.lstm3 = nn.LSTMCell(input_size= self.input_size3 , hidden_size = self.hidden_dim)

		#mixture density outputs
		self.e = nn.Linear(self.hidden_dim * 3, 1) # end of stroke probability
		self.pi = nn.Linear(self.hidden_dim * 3, self.n_gaussians) # mixture weights
		self.mu1 = nn.Linear(self.hidden_dim * 3, self.n_gaussians) #means - 1
		self.sig1 = nn.Linear(self.hidden_dim * 3, self.n_gaussians) #standard deviations - 1
		self.mu2 = nn.Linear(self.hidden_dim * 3, self.n_gaussians) #means -2
		self.sig2 = nn.Linear(self.hidden_dim * 3, self.n_gaussians) #standard deviations -2
		self.ro = nn.Linear(self.hidden_dim * 3, self.n_gaussians) # correlations

	def forward(self, x, char, char_mask, prev_state, prev_offset, prev_w):
		x = x.permute(1,0,2) # batch x seq_len x 3
		#drop_in = self.dropout(x)
		seq_len = x.shape[1]
		batch_size = x.shape[0]
		if prev_state is None:
			state = self.init_hidden(x.shape[0])
			h1, c1 = state[0]
			h2, c2 = state[1]
			h3, c3 = state[2]
		else:
			h1,c1,h2,c2,h3,c3 = prev_state
		if prev_w is None:
			prev_w = torch.ones(batch_size, self.vocab_size).cuda() # B x U
		else:
			prev_w = prev_w
		

		char_one_hot = one_hot(char, num_classes=self.vocab_size) # batchsize x char lenght x vocab size
		if self.use_cuda:
			x = x.cuda()
			char = char.cuda()
			
			char_mask = char_mask.cuda()
			char_one_hot = char_one_hot.cuda()
	
		char_len = char.shape[1]
		lstm1_out = torch.zeros(seq_len, batch_size, self.hidden_dim).cuda()
		lstm2_out = torch.zeros(seq_len, batch_size, self.hidden_dim).cuda()
		lstm3_out = torch.zeros(seq_len, batch_size, self.hidden_dim).cuda()
		#print(char_mask.shape)
		for i in range(seq_len): #calculate  wt
			#print(x.shape, x[:,i].shape, prev_w.shape)
			lstm1_input = torch.cat((x[:,i], prev_w), 1)
			
			h1, c1 = self.lstm1(lstm1_input, (h1, c1))
			
			# #window layer
			alpha, beta, kappa = torch.exp(self.alpha(h1)).unsqueeze(-1), torch.exp(self.beta(h1)).unsqueeze(-1), torch.exp(self.offset(h1)).unsqueeze(-1) # B x K 
			if prev_offset is None:
				prev_offset = kappa
			else:
				kappa = prev_offset + kappa

			prev_offset = kappa
			u = torch.arange(0, char_len).cuda()
			#print(kappa, u)
			phi = torch.sum(alpha * torch.exp(-beta * ((kappa - u) ** 2)), dim=1)
			#print('k',kappa, 'a', alpha, 'b', beta)
			#phi = phi * char_mask
			#phi = phi.masked_select(char_mask)
			#char_mask = self.batched_index_select_mask(phi, 1, char_mask)
			wt = torch.matmul(phi.unsqueeze(1), char_one_hot.float())
			wt = wt.squeeze(1)
			prev_w = wt
			
			
			h2, c2 = self.lstm2(torch.cat((x[:,i], h1, wt), 1), (h2, c2))
			h3, c3 = self.lstm3(torch.cat((x[:,i], h2, wt), 1), (h3, c3))
			lstm1_out[i] = h1
			lstm2_out[i] = h2
			lstm3_out[i] = h3 #seq_len x batchsize x hidden size
			
		#print(seq_len, kappa)
		lstm1_out = lstm1_out.permute(1, 0, 2) # batchsize x seq_len x hidden size
		lstm2_out = lstm2_out.permute(1, 0, 2)
		lstm3_out = lstm3_out.permute(1, 0, 2)
		lstm_out = torch.cat((lstm1_out, lstm2_out, lstm3_out), 2)
		#print(lstm_out)
		#mixture density layer
		e = self.e(lstm_out)
		e = 1 / (1 + torch.exp(e))
		#print(e)
		pi = self.pi(lstm_out) * (1 + self.bias)
		pi = torch.softmax(pi, -1)

		mu1 = self.mu1(lstm_out)
		mu2 = self.mu2(lstm_out)
		sig1 = self.sig1(lstm_out)
		sig1 = torch.exp(sig1 - self.bias)
		sig2 = self.sig2(lstm_out)
		sig2 = torch.exp(sig2 - self.bias)
		ro = self.ro(lstm_out)
		ro = torch.tanh(ro)
		prev_state = (h1,c1,h2,c2,h3,c3)
		
		return e, pi, mu1, mu2, sig1, sig2, ro, prev_state, phi,  prev_offset, prev_w
	

	def init_hidden(self, batch_size):
		state = []
		for i in range(self.num_layers):
			if self.use_cuda:
				hidden = (torch.zeros(batch_size, self.hidden_dim).cuda(), 
					torch.zeros(batch_size, self.hidden_dim).cuda())
			else:
				hidden = (torch.zeros(batch_size, self.hidden_dim).long(), 
					torch.zeros(batch_size, self.hidden_dim).long())
			state.append(hidden)
		return state

	def batched_index_select_mask(self, t, dim, inds):
		dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
		t = t.cuda()
		out = t * dummy# b x e x f
		return out
	def prediction_loss(self, e, pi, mu1, mu2, sig1, sig2, ro, target):
		#first we need to calculate the probability of p(x_t+1|y_t)
		# as we used #mixture gaussian, then we need to copy x1 or x2 to that dim
		target = target.permute(1,0,2)
		#target_mask = target_mask.permute(1,0)
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

	def multibivariate_sampling(self, pi, mu1, mu2, sig1, sig2, ro):
		#here can do batch sampling

		seq_len  = mu1.shape[0]
		#_, max_index = torch.max(pi, 1)
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
	def generate_samples(self, init_stroke, char, max_len):
		# char 1 x char_len
		prev_state = None
		prev_offset = None
		prev_w = None
		init_stroke = init_stroke.unsqueeze(0).unsqueeze(0).float().cuda() # 1 x 1 x 3
		char_mask = torch.ones_like(char)
		strokes = []
		for i in range(max_len):
			e, pi, mu1, mu2, sig1, sig2, ro, prev_state, phi, prev_offset, prev_w = self.forward(init_stroke, char, char_mask, prev_state, prev_offset, prev_w)
			
			e = e.squeeze(0)
			sample_mixture = self.multibivariate_sampling(pi.squeeze(0), mu1.squeeze(0), 
			mu2.squeeze(0), sig1.squeeze(0), sig2.squeeze(0), ro.squeeze(0))
			#print(e)
			e = Bernoulli(e)
			e = e.sample()
			
			init_stroke = torch.cat((e, sample_mixture.cuda()), 1) # 1 x 3
			
			
			strokes.append(init_stroke)
			
			init_stroke = init_stroke.unsqueeze(0)
			
			if phi.max(1)[1].item() > char.shape[1]-1: #exit
				break
		return torch.stack(strokes, 1)




if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	vocab_size = 77
	Model = ConditionalModel(use_cuda=use_cuda).cuda()
	ys = torch.randn(3, 4, 3).cuda()
	ys_mask = torch.ones(3, 4).cuda()
	text = torch.randint(0, vocab_size, (3, 4)).cuda()
	text_mask = torch.zeros_like(text).float()

	e, pi, mu1, mu2, sig1, sig2, ro = Model(ys, ys_mask, text, text_mask)

	loss = Model.prediction_loss(e, pi, mu1, mu2, sig1, sig2, ro, ys)
	print(loss.item())
	
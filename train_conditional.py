import argparse
import time
import torch
from utils import pad_list, plot_stroke
from dataset import Dataset, _collate_fn
from torch.utils.data import DataLoader
from models.conditional import ConditionalModel
from matplotlib import pyplot as plt
import numpy as np
import pickle
parser = argparse.ArgumentParser(
    "HandWriting generation.")
parser.add_argument('--n_layers', default=3, type=int,
                    help='Number of LSTM layers')
parser.add_argument('--input_dim', default=3, type=int,
                    help='Number of input diminension')
parser.add_argument('--hidden_dim', default=400, type=int,
                    help='hidden layer size')
parser.add_argument('--num-gaussian', default=20, type=int,
                    help='Number of gaussian components')
parser.add_argument('--dropout', default=3, type=float,
                    help='dropout rate')
parser.add_argument('--batch_size', default=20, type=int,
                    help='batch size')
parser.add_argument('--data_path', default='data/',
                    help='data')
parser.add_argument('--num_epoch', default=100, type=int,
                    help='traing epoch')
parser.add_argument('--lr', default=0.0005, type=float,
                    help='learning rate')
parser.add_argument('--k_gaussian', default=10, type=int,
                    help='k gaussian')


def train(args):
	dataset = Dataset(args.data_path, args.batch_size)
	data_loader = DataLoader(dataset, batch_size=1, 
		shuffle=True, collate_fn=_collate_fn)
	outputlayer_name = ['e', 'pi', 'mu1', 'mu2', 'sig1', 'sig2', 'ro'] # for gradient cliping
	pkl_file = open('char2int.pkl', 'rb')
	char2int = pickle.load(pkl_file)
	pkl_file.close()
	test_char = "welcome to lyrebird"
	char2array = torch.from_numpy(np.array([char2int[x] for x in test_char])).long().cuda()
	char2array = char2array.unsqueeze(0)
	epochs = args.num_epoch
	lr = args.lr
	max_len = 600
	use_cuda = torch.cuda.is_available()
	print_freq = 10
	if use_cuda:
		Model = ConditionalModel(use_cuda=use_cuda).cuda()
	else:
		Model = ConditionalModel()
	#train

	optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
	plot_loss = []
	for epoch in range(epochs):
		#batch_loss = 0
		total_loss = 0
		start = time.time()
		for i, (data) in enumerate(data_loader):
			ys, char, ys_mask, char_mask = data
			if use_cuda:
				ys = ys.cuda()
				char = char.long().cuda()
				ys_mask = torch.from_numpy(ys_mask).long().cuda()
				char_mask = torch.from_numpy(char_mask).long().cuda()
			#print(ys)
			prev_state, prev_offset, prev_w = None, None, None
			e, pi, mu1, mu2, sig1, sig2, ro, _, _, _, _= Model(ys.permute(1,0,2)[:-1], char, char_mask, prev_state, prev_offset, prev_w)
			loss = Model.prediction_loss(e, pi, mu1, mu2, sig1, sig2, ro, ys.permute(1,0,2)[1:])
			optimizer.zero_grad()
			loss.backward()

			for name, param in Model.named_parameters():
				if 'lstm' in name:
					param.grad.data.clamp_(-10, 10)
				else:
					param.grad.data.clamp_(-100, 100)
			optimizer.step()
			plot_loss.append(loss.item())
			total_loss += loss.item()
			#print(data)
			if i % print_freq == 0:
				stroke = torch.tensor([1, 0, 0])
				print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

				prediction= Model.generate_samples(stroke, char2array, max_len)
				prediction = prediction.squeeze(0).cpu().numpy()
				print(prediction.shape)
				plot_stroke(prediction, 'generation_fix.png')
				torch.save(Model.state_dict(), "./checkpoint/synthesis_model")

if __name__ == '__main__':
	args = parser.parse_args()
	train(args)

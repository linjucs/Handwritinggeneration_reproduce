import argparse
import time
import torch
from utils import pad_list, plot_stroke
from dataset import Dataset, _collate_fn
from torch.utils.data import DataLoader
from models.unconditional import UnconditionalModel
from matplotlib import pyplot as plt

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
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')

def train(args):
	dataset = Dataset(args.data_path, args.batch_size)
	data_loader = DataLoader(dataset, batch_size=1, 
		shuffle=True, collate_fn=_collate_fn)
	outputlayer_name = ['e', 'pi', 'mu1', 'mu2', 'sig1', 'sig2', 'ro'] # for gradient cliping
	epochs = args.num_epoch
	lr = args.lr
	max_len = 800
	use_cuda = torch.cuda.is_available()
	print_freq = 10
	if use_cuda:
		Model = UnconditionalModel(use_cuda=use_cuda).cuda()
	else:
		Model = UnconditionalModel()
	#train
	
	optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
	plot_loss = []
	for epoch in range(epochs):
		#batch_loss = 0
		total_loss = 0
		start = time.time()
		for i, (data) in enumerate(data_loader):

			if use_cuda:
				ys ,_, ys_mask, _= data
				ys = ys.cuda()
				ys_mask = torch.tensor(ys_mask).cuda()
			prev_state = None
			#print(ys)
			e, pi, mu1, mu2, sig1, sig2, ro, _ = Model(ys.permute(1,0,2)[:-1], prev_state)
			
			loss = Model.prediction_loss(e, pi, mu1, mu2, sig1, sig2, ro, ys.permute(1, 0 ,2)[1:], ys_mask.permute(1,0)[1:])
			optimizer.zero_grad()
			loss.backward()
			# follow the paper setting
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
				print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
				init_stoke = torch.tensor([1, 0, 0]).float().cuda()
				generation = Model.convert2strokes(ys[0][0], max_len)
				plot_stroke(generation.squeeze(0).cpu().numpy(), 'generation.png')
				torch.save(Model.state_dict(), "./checkpoint/prediction_model")

if __name__ == '__main__':
	args = parser.parse_args()
	train(args)

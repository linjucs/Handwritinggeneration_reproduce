import numpy as np
import torch
import torch.utils.data as data

from utils import pad_list
import pickle

class Dataset(data.Dataset):
	def __init__(self, data_path, batch_size):
		super(Dataset, self).__init__()
		self.data_path = data_path
		strokes = np.load(self.data_path + '/' + 'strokes-py3.npy', allow_pickle=True)
		characters = set()
		with open(self.data_path + '/' + 'sentences.txt') as f:
			texts = f.readlines()

		assert len(strokes) == len(texts)
		#count characters
		for line in texts:
			characters.update(line)
		vocob_szie = len(characters)
		# the number of total characters is 78
		char2int = dict([(elem, i) for i, elem in enumerate(characters)])
		f = open("char2int.pkl","wb")
		pickle.dump(char2int,f)
		f.close()
		texts_int = []
		for line in texts:
			texts_int.append([char2int[x] for x in line])
		sorted_paired = [(stroke, text) for stroke, text in sorted(zip(strokes,texts_int), key=lambda pair: len(pair[0]))]
		#sort it by lengths
		minibatch = []
		start = 0

		while True:
			end = min(len(sorted_paired), start + batch_size)

			minibatch.append(sorted_paired[start:end])

			if end >= len(sorted_paired):
				break
			start = end
		self.minibatch = minibatch

	def __getitem__(self, index):
		return self.minibatch[index]

	def __len__(self):
		return len(self.minibatch)



def _collate_fn(batch):
# as do the minibatch already in dataset, so here batch size is 1
	assert len(batch) == 1
	ys, xs = zip(*batch[0])
	ys_pad, ys_mask = pad_list([torch.from_numpy(y) for y in ys], 0)
	xs_pad, xs_mask = pad_list([torch.from_numpy(np.array(x)) for x in xs], 0)
	return ys_pad, xs_pad, ys_mask, xs_mask

if __name__ == '__main__':
	data_path = './data'
	dataset = Dataset(data_path, 20)
	data_loader = data.DataLoader(dataset, batch_size=1, num_workers=3,
                                shuffle=True, collate_fn=_collate_fn)
	for i, (data) in enumerate(data_loader):
		ys, xs, ys_mask, xs_mask = data
		print(ys_mask.shape, xs_mask.shape)
		

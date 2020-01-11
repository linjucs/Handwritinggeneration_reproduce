import numpy
import sys
import torch
import pickle
from models.unconditional import UnconditionalModel
from models.conditional import ConditionalModel

def generate_unconditionally(random_seed=1):
    # Input:
    #   random_seed - integer
    numpy.random.seed(random_seed)
    use_cuda = torch.cuda.is_available()
    seq_len = 700
    if use_cuda:
        model = UnconditionalModel(use_cuda=use_cuda).cuda()
    else:
        model = UnconditionalModel(use_cuda=use_cuda)
    
    model.load_state_dict(torch.load('../checkpoint/prediction_model'))
    
    es = numpy.random.binomial(1, 0.8, 1)
    stroke = numpy.random.multivariate_normal([0.5, 0.5], [[0.1,-0.1],[-0.1,0.3]], 1)
    
    init_stroke = numpy.concatenate([numpy.expand_dims(es, axis=0),stroke], 1)
    
    init_stroke = torch.tensor(init_stroke).float().cuda()

    stroke = model.generate_samples(init_stroke.squeeze(), seq_len)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke.squeeze(0).cpu().numpy()


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer
    numpy.random.seed(random_seed)
    use_cuda = torch.cuda.is_available()
    seq_len = 600
    if use_cuda:
        model = ConditionalModel(use_cuda=use_cuda).cuda()
    else:
        model = ConditionalModel(use_cuda=use_cuda)
    
    model.load_state_dict(torch.load('../checkpoint/synthesis_model'))
   
    es = numpy.random.binomial(1, 0.8, 1)
    stroke = numpy.random.multivariate_normal([0.5, 0.5], [[0.1,-0.1],[-0.1,0.3]], 1)
    
    init_stroke = numpy.concatenate([numpy.expand_dims(es, axis=0),stroke], 1)
    
    init_stroke = torch.tensor(init_stroke).float().cuda()
    #init_stroke = torch.tensor([1, 0, 0])
    pkl_file = open('../char2int.pkl', 'rb')
    char2int = pickle.load(pkl_file)
    #print(char2int)
    pkl_file.close()
    char2array = torch.from_numpy(numpy.array([char2int[x] for x in text])).long().cuda()
    char2array = char2array.unsqueeze(0)
    stroke = model.generate_samples(init_stroke.squeeze(), char2array, seq_len)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke.squeeze(0).cpu().numpy()


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'

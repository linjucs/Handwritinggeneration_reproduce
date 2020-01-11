import numpy
strokes = numpy.load('../data/strokes-py3.npy', allow_pickle=True)
stroke = strokes[0]
import torch

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
    model.eval()
    model = torch.load('../checkpoint/prediction_model')
    es = numpy.random.binomial(1, 0.8, 1)
    stroke = x = numpy.random.multivariate_normal([0.5, 0.5], [[0.1,-0.1],[-0.1,0.3]], 1)
    print(es, stroke)
    init_stroke = numpy.concatenate([numpy.expand_dims(es, axis=0),stroke], 1)
    print(init_stroke.shape) 
    init_stroke = torch.tensor(init_stroke)
    stroke = model(init_stroke, seq_len)
    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke.squeeze(0).cpu().numpy()


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
stroke = generate_unconditionally()
plot_stroke(stroke)

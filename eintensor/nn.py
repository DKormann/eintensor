from tinygrad import nn
from tinygrad.nn import optim
from eintensor import EinTensor, EinDim

class Adam:
  def __init__(self, weights: list[EinTensor], lr = 0.001):
    for w in weights: w.requires_grad = True
    self.adam = optim.Adam([w.data for w in weights]
    , lr = lr)
  def step(self): return self.adam.step()

def mnist():
  xtrain, ytrain, xtest, ytest = nn.datasets.mnist()
  return (EinTensor.fromTensor(xtrain.squeeze(), 'Ntrain', 'width', 'height'),
    EinTensor.fromTensor(ytrain, 'Ntrain'),
    EinTensor.fromTensor(xtest.squeeze(), 'Ntest', 'width', 'height'),
    EinTensor.fromTensor(ytest, 'Ntest'))


def get_parameters(model):
  return list(nn.state.get_state_dict(model, '', EinTensor).values())
  




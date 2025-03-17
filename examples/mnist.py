from eintensor import EinTensor, EinDim
import matplotlib.pyplot as plt
from eintensor.nn import mnist
from eintensor.nn import Adam

import tinygrad

trainx, trainy, testx, testy = mnist()
Nsamples, width, height = trainx.einshape

Classes = EinDim("NClasses", 10)

class NN:
  def __init__(self):
    hdim = EinDim('hidden', 400)
    self.w1 = EinTensor.rand(width, height, hdim, requires_grad = True) * 0.01
    self.w2 = EinTensor.rand(hdim, Classes, requires_grad = True) * 0.01

  # @tinygrad.TinyJit
  def forward(self, x):
    x = (self.w1 @ x).relu()
    x = (self.w2 @ x)
    x = x.softmax(Classes)
    return x

nn = NN()

opt = Adam([
  nn.w1,
  nn.w2
], lr = 0.01)

def step():
  p = nn.forward(trainx)
  loss = p.sparse_categorical_crossentropy(trainy)
  loss.backward()
  opt.step()
  return loss.numpy()

EinTensor.settrain(True)

for i in range(10):
  print(step())


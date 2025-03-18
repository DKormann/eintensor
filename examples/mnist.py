from eintensor import EinTensor, EinDim
import matplotlib.pyplot as plt
from eintensor.nn import mnist
from eintensor.nn import Adam
from tinygrad import TinyJit
import time

trainx, trainy, testx, testy = mnist()
Nsamples, width, height = trainx.einshape

Classes = EinDim("NClasses", 10)

class NN:
  def __init__(self):
    hdim = EinDim('hidden', 200)
    eps = 1/hdim.size
    self.w1 = EinTensor.rand(width, height, hdim) * eps
    self.w2 = EinTensor.rand(hdim, Classes) * eps

  def forward(self, x):
    x = x.float()/255.
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

  return p, loss.numpy()

EinTensor.settrain(True)

jitstep = TinyJit(step)

for i in range(1000):
  if not i or (i+1) %10 == 0:
    p, res = jitstep()
    pred = p.argmax_to(Nsamples)

    vpred = nn.forward(testx).argmax_to(testy) == testy
    
    print(i, "acc:",(pred == trainy).mean().numpy())


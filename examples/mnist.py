from eintensor import EinTensor, EinDim
import matplotlib.pyplot as plt
from eintensor.nn import mnist
from eintensor.nn import Adam
import eintensor
from tinygrad import TinyJit
import time

trainx, trainy, testx, testy = mnist()
Nsamples, width, height = trainx.einshape

Classes = EinDim("NClasses", 10)

class NN:
  def __init__(self):
    hdim = EinDim('hidden', 400)

    self.w1 = EinTensor.randn(width, height, hdim)
    self.w1 /= self.w1.einshape.numel
    self.w2 = EinTensor.randn(hdim, Classes)
    self.w2 /= self.w2.einshape.numel

  def forward(self, x):
    x = x.float()/255.
    x = (self.w1 @ x).relu()
    x = (self.w2 @ x)
    x = x.softmax(Classes)
    return x

nn = NN()


opt = Adam(
  eintensor.nn.get_parameters(nn)
  
  , lr = 0.001)

BatchDim = EinDim('Batch', 100)


def step():

  idx = EinTensor.randint(BatchDim, high=Nsamples.size)
  x = trainx[idx]
  y = trainy[idx]
  p = nn.forward(x)
  loss = p.sparse_categorical_crossentropy(y)
  loss.backward()
  opt.step()

  return p, loss.numpy()

EinTensor.train(True)
jitstep = TinyJit(step)

from tinygrad import Tensor


@TinyJit
def test():
  p = nn.forward(testx)
  acc = p.argmax(Classes).eq(testy).mean()
  return acc


it = 400

for i in range(it):
  p, loss = jitstep()

  if not i or i % (it//10) ==0:
    acc = test().item()
    print(f"epoch {i} {acc=} loss {loss}")


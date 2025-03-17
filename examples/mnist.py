from eintensor import EinTensor, EinDim
import matplotlib.pyplot as plt
from tinygrad.nn.datasets import mnist
from tinygrad import dtypes

trainx, trainy, testx, testy = mnist()

trainx = EinTensor.fromTensor(trainx, "Nsamples", -1, 'width', 'height')

Nsamples, _, width, height = trainx.einshape

trainy = EinTensor.fromTensor(trainy, Nsamples)

trainx = trainx.reshape(Nsamples,width,height)

Classes = EinDim("NClasses", 10)

class NN:
  def __init__(self):
    self.w = EinTensor.rand(width, height, Classes)
  
  def forward(self, x):
    return self.w @ x

nn = NN()

p = nn.forward(trainx)


from tinygrad import nn, Tensor


loss = p.sparse_categorical_crossentropy(trainy)

print(loss)
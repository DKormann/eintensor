import unittest
from tinsor import Shape, Tensor

A, B, C, D = Shape(A=3, B=4, C=5, D=6)
class TestTensor(unittest.TestCase):
  def test_tensor(self):

    x = Tensor.rand(A, B)
    y = Tensor.rand(B, C)

    assert x.shape == Shape(A=3, B=4)
  
  def test_binary_ops(self):
    x = Tensor.rand(A, B)
    y = Tensor.rand(B, C)

    z = x @ y
    assert z.shape == Shape(A=3, C=5)

    z = x * y
    assert z.shape == Shape(A=3, B=4, C=5)
  
  def test_reduce_ops(self):
    x = Tensor.rand(A, B, C)
    y = x.sum(B)
    assert y.shape == Shape(A=3, C=5)

  def test_numpy(self):
    x = Tensor.rand(A, B)
    assert x.numpy().shape == (3, 4)
    assert x.numpy().dtype == 'float32'

  def test_movement_ops(self):
    x = Tensor.rand(A, B)
    y = x.permute(B, A)
    assert y.shape == Shape(B, A)
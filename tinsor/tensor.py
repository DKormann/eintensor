from dataclasses import dataclass

from typing import Any
import tinygrad
import tinygrad.tensor
from tinygrad.helpers import dedup

@dataclass(eq=True, frozen=True)
class Dim:
  name: str
  size: int

class Shape():
  dims: tuple[Dim]

  def __init__(self,*dims,**kwargs):
    dims = tuple(Dim(f"_{d}", d) if isinstance(d, int) else d for d in dims)
    self.dims = dims + tuple(Dim(k, v) for k,v in kwargs.items())
  
  def __repr__(self): return f'Shape({", ".join([f"{d.name}={d.size}" for d in self.dims])})'
  def __iter__(self): return (d for d in self.dims)

  def __eq__(self, other):
    return self.dims == other.dims
  
  @property
  def size(self): return tuple([d.size for d in self.dims])

  @property
  def keys(self): return [d.name for d in self.dims]


class Tensor:
  shape: Shape
  tensor: tinygrad.Tensor

  def __init__(self, shape: Shape, tensor: tinygrad.Tensor):
    self.shape = shape
    self.tensor = tensor

  def numpy(self): return self.tensor.numpy()

  def __repr__(self): return f'<Tensor {self.shape} {self.tensor.dtype} device={self.tensor.device}>'

  def __add__(self, other: 'Tensor'):
    return Tensor(self.shape, self.tensor + other.tensor)

  def ones(*dims): return Tensor((shp:=Shape(*dims)), tinygrad.Tensor.ones(shp.size))
  def zeros(*dims): return Tensor((shp:=Shape(*dims)), tinygrad.Tensor.zeros(shp.size))
  def rand(*dims): return Tensor((shp:=Shape(*dims)), tinygrad.Tensor.rand(shp.size))
  def eye(dim): return Tensor((shp:=Shape(dim, dim)), tinygrad.Tensor.eye(shp.size[0]))

  # unary ops
  def __neg__(self): return Tensor(self.shape, -self.tensor)
  def sigmoid(self): return Tensor(self.shape, self.tensor.sigmoid())
  def relu(self): return Tensor(self.shape, self.tensor.relu())

  # binary ops
  def binary_fn(self, other: 'Tensor', fn: Any) -> 'Tensor':
    newshape = Shape(*dedup(self.shape.dims + other.shape.dims))

    stensor = self.tensor
    otensor = other.tensor

    for k in newshape.keys:
      if k not in self.shape.keys: stensor = stensor.unsqueeze(-1)
      if k not in other.shape.keys: otensor = otensor.unsqueeze(-1)

    last = len(newshape.keys)
    perm = [other.shape.keys.index(k) if k in other.shape.keys else (last:=last-1) for k in newshape.keys]
    otensor = otensor.permute(*perm)

    last = len(newshape.keys)
    perm = [self.shape.keys.index(k) if k in self.shape.keys else (last:=last-1) for k in newshape.keys]
    stensor = stensor.permute(*perm)

    return Tensor(newshape, fn(stensor, otensor))
  def __add__(self, other: 'Tensor'): return self.binary_fn(other, tinygrad.Tensor.add)
  def __sub__(self, other: 'Tensor'): return self.binary_fn(other, tinygrad.Tensor.sub)
  def __mul__(self, other: 'Tensor'): return self.binary_fn(other, tinygrad.Tensor.mul)
  def __truediv__(self, other: 'Tensor'): return self.binary_fn(other, tinygrad.Tensor.div)
  def __matmul__(self, other: 'Tensor')->'Tensor':
    reduce_dim = [d for d in self.shape.dims if d.name in other.shape.keys][0]
    return (self * other).sum(reduce_dim)

  # reduce ops
  @staticmethod
  def reduce_fn(fn):
    def reduce(self:"Tensor", *axes: Dim):
      axes = [self.shape.keys.index(k.name) for k in axes]
      return Tensor(Shape(*[d for i,d in enumerate(self.shape.dims) if i not in axes]), fn(self.tensor, axes))
    return reduce
  
  sum = reduce_fn(tinygrad.Tensor.sum)
  max, min = map(reduce_fn, [tinygrad.Tensor.max, tinygrad.Tensor.min])
  argmax, argmin, mean, softmax = map(reduce_fn, [tinygrad.Tensor.argmax, tinygrad.Tensor.argmin, tinygrad.Tensor.mean, tinygrad.Tensor.softmax])

  def permute(self, *dims: Dim):
    for d in dims: assert d in self.shape.dims, f'{d} not in {self.shape}'
    perm = [self.shape.keys.index(d.name) for d in dims]
    return Tensor(Shape(*dims), self.tensor.permute(*perm))
  
  # @property
  # def T(self): return self.transpose()

  def transpose(self, dim1 = None, dim2 = None):
    dim1, dim2 = dim1 or 1, dim2 or 0
    order = list(self.shape.dims)
    order[dim1], order[dim2] = order[dim2], order[dim1]
    return self.permute(*order)

  def __getattr__(self, name):
    if self.shape.keys[0] == name: return self
    idx = [i for i,d in enumerate(self.shape) if d.name == name][0]
    return self.transpose(idx)


  def expand(self, *dims: Dim):
    dims = tuple(d if d not in self.shape.dims else Dim("_"+d.name,d.size) for d in dims)
    tensor = self.tensor
    for d in dims: tensor = tensor.unsqueeze(-1)
    return Tensor(Shape(self.shape.dims + dims), tensor)



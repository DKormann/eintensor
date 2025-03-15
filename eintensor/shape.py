from tinygrad import Tensor
from tinygrad.ops import SimpleMathTrait
import math
_dimensions = {}

class EinDim:
  def __init__ (self,name:str, size:int):
    if name in _dimensions:
      assert _dimensions[name] == size, f"Dimension {name} already exists with size {_dimensions[name]}"
    _dimensions[name] = size
    self.name = name
    self.size = size
  
  def __repr__(self): return self.name
  def __eq__(self, other):
    
    return self.size == other.size and self.name == other.name
  def __hash__(self): return hash(self.name)
  def copy(self):
    ctr=1
    newname = f'{self.name}_{ctr}'
    while newname in _dimensions: newname = f'{self.name}_{ctr:=ctr+1}'
    return EinDim(newname, self.size)

I = EinDim('1', 1)


class einShape:

  def __init__(self, *dims):
    dims = tuple( EinDim(str(d), d)  if isinstance(d, int) else d for d in dims)
    self.dims = dims
    self.shape = tuple(map(lambda x: x.size, dims))
    self.numel = math.prod(self.shape)

  def __repr__(self):return f'einShape{self.dims}'

  def permute(self, order):
    assert len(order) == len(self.dims)
    return einShape(*[self.dims[i] for i in order])

  def union (self, *others): return einShape(*list(dict.fromkeys(sum([d.dims for d in [self, *others]], tuple()))))
  def inter (self, *others): return einShape(*[d for d in self.dims if all(d in o.dims for o in others)])

  def add(self, *dims): return einShape(*list(dict.fromkeys(self.dims + dims)))
  def remove(self, *dims): return einShape(*[d for d in self.dims if d not in dims])

  def __eq__(self, other): return self.dims == other.dims

tensor_att = ['dtype', 'lazydata', 'numpy', 'shape']
reduce_ops = ['sum', 'mean', 'max', 'min', 'prod', 'std', 'var']

def create_generatator(fn):
  def wrapped(*shape, **kwargs):
    einshape = einShape(*shape)
    return EinTensor(einshape, fn(*einshape.shape, **kwargs))
  return wrapped


fns = []


class EinTensor():
  def __init__(self, shape:einShape, data:Tensor):
    assert shape.shape == data.shape
    self.einshape = shape
    self.data = data

  ones = create_generatator(Tensor.ones)
  zeros = create_generatator(Tensor.zeros)
  rand = create_generatator(Tensor.rand)
  randn = create_generatator(Tensor.randn)
  randint = create_generatator(Tensor.randint)

  arange = create_generatator(Tensor.arange)

  def linspace(start, end, dim): return EinTensor(einShape(dim), Tensor.linspace(start, end, dim.size))

  def stack(*tensors, dim:EinDim = None):
    l = len(tensors)
    if dim is None: dim = EinDim(f'StackDim:{l}', l)
    assert dim.size == l, f"Dimension {dim} must have size {l}"
    newshape = einShape.union(*[t.einshape for t in tensors])
    tds = [t.expand(newshape).data.expand(newshape.shape) for t in tensors]
    return EinTensor(einShape(*((dim,) + newshape.dims)), Tensor.stack(*tds))
  
  # def concat(self, other:EinTensor, selfdim, otherdim = None, newdim = None):
  #   if otherdim is None: otherdim = selfdim
  #   catsize = selfdim.size + otherdim.size
  #   if newdim is None: newdim = EinDim(f"ConcatDim_{catsize}", catsize)
  #   assert newdim.size == catsize, f"Dimension {newdim} must have size {catsize}"
  #   assert selfdim in self.einshape.dims, f"Dimension {selfdim} not found in {self.einshape}"
  #   assert otherdim in other.einshape.dims, f"Dimension {otherdim} not found in {other.einshape}"
  #   restidms = einShape.union(self.einshape.remove(selfdim), other.einshape.remove(otherdim))
  #   self = self.expand(restidms.add(selfdim))
  #   other = other.expand(restidms.add(otherdim))
  #   # return EinTensor(restidms.add(newdim)

  def __repr__(self): return f'<einTensor [{', '.join(map(lambda x: x.name, self.einshape.dims))}]>'

  def reshape(self, einshape:einShape):
    assert self.einshape.numel == einshape.numel, f"Cannot reshape {self.einshape} to {einshape}, {self.einshape.numel} != {einshape.numel}"
    return EinTensor(einshape, self.data.reshape(einshape.shape))

  def expand(self, newshape):
    sparse_shape = einShape(*(self.einshape.dims + (I,) * (len(newshape.dims) - len(self.einshape.dims))))
    reshaped = self.reshape(sparse_shape)

    extra = len(self.einshape.dims)-1
    order = [self.einshape.dims.index(k) if k in self.einshape.dims else (extra := extra + 1) for k in newshape.dims]
    return EinTensor(reshaped.einshape.permute(order), reshaped.data.permute(order))
  
  def __getattribute__(self, name):
    if name in tensor_att:
      return getattr(self.data, name)
    return super().__getattribute__(name)

def create_elementwise(fn):
  def wrapped (one:EinTensor, other):
    if (type(other) == int or type(other) == float):
      val = other
      other = EinTensor.ones()
      other.data *= val
    shape = one.einshape.union(other.einshape)
    return EinTensor(
      one.einshape.union(other.einshape),
      fn(one.expand(shape).data, other.expand(shape).data))
  return wrapped

binary_ops = [op for op in SimpleMathTrait.__dict__ if op not in EinTensor.__dict__]

for op in binary_ops:
  fn = getattr(SimpleMathTrait, op)
  setattr(EinTensor, op, create_elementwise(fn))

unary_ops = ['__neg__', 'abs', '__invert__', 'float', 'int', 'bool']
for op in unary_ops: setattr(EinTensor, op, (lambda name: lambda x: EinTensor(x.einshape, getattr(x.data, name)()))(op))

def create_reduce(fn, inverse = False):
  def wrapped (x, *axes:tuple[EinDim]):
    if axes == []: axes = x.einshape.dims
    if inverse: axes = [d for d in x.einshape.dims if d not in axes]
    return EinTensor(einShape(*[k for k in x.einshape.dims if k not in axes]),
      fn(x.data, [x.einshape.dims.index(k) for k in axes]))
  return wrapped

for op in reduce_ops:
  setattr(EinTensor, op, create_reduce(getattr(Tensor, op)))
  setattr(EinTensor, f'{op}_to', create_reduce(getattr(Tensor, op), True))

if __name__ == '__main__':
  K,V,S,T = EinDim('K', 3), EinDim('V', 5), EinDim('S', 7), EinDim('T', 11)

  Dim, Nsamples, Out = EinDim('Dim', 10), EinDim('Nsamples', 100), EinDim('Out', 20)

  x = EinTensor.ones(Nsamples, Dim)
  w = EinTensor.ones(Dim, Out)

  print(x.stack(w))
  print(x.stack(EinTensor.ones(Dim, Nsamples)))




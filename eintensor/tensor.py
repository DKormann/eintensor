from tinygrad import Tensor
from tinygrad.ops import SimpleMathTrait
import math

_dimensions = {}

class EinDim:
  def __init__ (self,name:str, size:int=None):
    if size == None:
      assert type(name) == int, f'EinDim requires at least an int for size'
      size = name
      name = f'_{size}'
    if name in _dimensions:
      assert _dimensions[name] == size, f"Dimension {name} already exists with size {_dimensions[name]}"
    _dimensions[name] = size
    self.name = name
    self.size = size
  
  def __repr__(self): return self.name
  def __eq__(self, other):
    
    return isinstance(other, EinDim) and self.size == other.size and self.name == other.name
  def __hash__(self): return hash(self.name)
  def copy(self):
    ctr=1
    newname = f'{self.name}_{ctr}'
    while newname in _dimensions: newname = f'{self.name}_{ctr:=ctr+1}'
    return EinDim(newname, self.size)


I = EinDim( 1)


class einShape:

  def __init__(self, *dims):
    dims = tuple( EinDim(str(d), d)  if isinstance(d, int) else d for d in dims)
    self.dims = dims
    self.shape = tuple(map(lambda x: x.size, dims))
    assert all(type(s)==int for s in self.shape)
    self.numel = math.prod(self.shape)

  def __repr__(self):return f'einShape{self.dims}'

  def permute(self, order):
    assert len(order) == len(self.dims)
    return einShape(*[self.dims[i] for i in order])

  def __iter__(self):
    return (self.dims).__iter__()

  def union (self, *others): return einShape(*list(dict.fromkeys(sum([d.dims for d in [self, *others]], tuple()))))
  def inter (self, *others): return einShape(*[d for d in self.dims if all(d in o.dims for o in others)])

  def broadcast(*shapes):
    defs = [d for d in shapes if d is not None]
    if len(defs) == 0: return shapes
    big = defs[0].inter(*defs[1:])
    return [big if d is None else d for d in shapes]

  def add(self, *dims): return einShape(*list(dict.fromkeys(self.dims + dims)))
  def diff(self, *dims): return einShape(*[d for d in self.dims if d not in dims])

  def __eq__(self, other): return self.dims == other.dims




tensor_att = ['dtype', 'device', 'lazydata', 'numpy', 'item', 'tolist', 'shape', 'requires_grad', "training"]
tensor_fns = ['backward']
reduce_ops = ['sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'prod', 'std', 'var', 'all', 'any']
unary_ops = ['__neg__', 'abs', '__invert__', 'float', 'int', 'bool', 'sqrt', "relu", 'exp']


def create_generatator(fn):
  def wrapped(*shape, **kwargs):
    einshape = einShape(*shape)
    return EinTensor(einshape, fn(*einshape.shape, **kwargs))
  return wrapped

fns = []


class EinTensor():

  def fromTensor(tensor:Tensor, *dims):
    dims = dims + (-1,) * len(tensor.shape)
    dims = [EinDim(t) if d==-1 else EinDim(d,t) if type(d) == str else d for [t,d] in zip(tensor.shape, dims)]

    assert all(type(d) == EinDim for d in dims)
    return EinTensor(dims, tensor)
  


  def __init__(self, shape:einShape, data:Tensor):
    if not isinstance(shape, einShape): shape = einShape(*shape)
    if not isinstance(data, Tensor): data = Tensor(data)
    assert shape.shape == data.shape, f"Shape {shape.shape} does not match data shape {data.shape}"
    self.einshape:einShape = shape
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
    tds = [t.expand(*newshape.dims).data.expand(newshape.shape) for t in tensors]
    return EinTensor(einShape(*((dim,) + newshape.dims)), Tensor.stack(*tds))
  
  def __repr__(self): return f"<einTensor [{', '.join(map(lambda x: x.name, self.einshape.dims))}]>"

  def reshape(self, *einshape:tuple[EinDim]):
    einshape = einShape(*einshape)
    assert self.einshape.numel == einshape.numel, f"Cannot reshape {self.einshape} to {einshape}, {self.einshape.numel} != {einshape.numel}"
    return EinTensor(einshape, self.data.reshape(einshape.shape))

  def expand(self, *dims:tuple[EinDim]):
    newshape = einShape(*dims)
    sparse_shape = (self.einshape.dims + (I,) * (len(newshape.dims) - len(self.einshape.dims)))
    reshaped = self.reshape(*sparse_shape)
    extra = len(self.einshape.dims)-1
    order = [self.einshape.dims.index(k) if k in self.einshape.dims else (extra := extra + 1) for k in newshape.dims]
    return EinTensor(newshape, reshaped.data.permute(order).expand(newshape.shape))
  
  def permute(self, *dims:EinDim):
    assert len(dims) == len(self.einshape.dims)
    order = [self.einshape.dims.index(k) for k in dims]
    return EinTensor(self.einshape.permute(order), self.data.permute(order))

  def wrap_tensor(self, arg):
    if isinstance(arg, Tensor):
      if arg.shape == self.shape: return EinTensor(self.einshape, arg)
      return EinTensor.fromTensor(arg)
    return arg

  def __setattribute__( name, value):
    print(name, value)
    if name == 'training': self.data.training = value
    else: super().__setattr__(name, value)

  def __getattribute__(self, name):
    if name in tensor_att:
      return self.wrap_tensor(getattr(self.data, name))
    if name in tensor_fns:
      fn = getattr(self.data, name)
      def wrapper(*args, **kwargs):
        return self.wrap_tensor(fn(*args, **kwargs))
      return wrapper

    return super().__getattribute__(name)

  def sparse_categorical_crossentropy(self, y):
    self = self.permute(*y.einshape.inter(self.einshape).dims, *self.einshape.diff(*y.einshape.dims).dims)
    return EinTensor([], self.data.sparse_categorical_crossentropy(y.data))

  def softmax(self, dim:EinDim = None):
    if dim is None: dim = -1
    else: dim = self.einshape.dims.index(dim)
    return self.wrap_tensor(self.data.softmax(dim))

  def clamp(self, min = None, max = None):
    if min is not None: self = self.maximum(min)
    if max is not None: self = self.minimum(max)
    return self
  
  def backward(self): return EinTensor(self.shape, self.data.backward())

  @property
  def grad(self):
    g = self.data.grad
    return g if g is None else EinTensor(self.einshape, g)

  def __matmul__(self, other): return (self * other).sum(*self.einshape.inter(other.einshape).dims)

  def __getitem__(self, index):
    if not isinstance(index,tuple):index = (index,)

    def parsedims(dims, index):
      if index == (): return dims
      if index[0] is None: return (I, *parsedims(dims, index[1:]))
      if type(index[0]) == int: return parsedims(dims[1:], index[1:])
      if type(index[0]) == slice: return (None,) + parsedims(dims[1:], index[1:])
      if type(index[0]) == EinTensor: return parsedims(dims[1:], index[1:])
      raise ValueError(f"Invalid index {index}")

    newdata = self.data.__getitem__(tuple(i.data if isinstance(i, EinTensor) else i for i in index))
    big = einShape().union(*[i.einshape for i in index if isinstance(i, EinTensor)])

    newdims = big.dims+ parsedims(self.einshape.dims, tuple(i for i in index))
    newdims = [EinDim(a) if b is None else b if b.size == a else EinDim(a) for [a,b] in zip(newdata.shape, newdims)]

    return EinTensor(einShape(*newdims), newdata)

  @staticmethod
  def train(x=True): Tensor.training = x
  

def create_elementwise(fn):
  def wrapped (one:EinTensor, other):
    if (type(other) == int or type(other) == float):
      val = other
      other = EinTensor.ones()
      other.data *= val
    shape = one.einshape.union(other.einshape)
    return EinTensor(
      one.einshape.union(other.einshape),
      fn(one.expand(*shape.dims).data, other.expand(*shape.dims).data))
  return wrapped

binary_ops = [op for op in SimpleMathTrait.__dict__ if op not in EinTensor.__dict__] \
  + ['__pow__', 'minimum', 'maximum', "__eq__"]

for op in binary_ops:
  fn = getattr(Tensor, op)
  setattr(EinTensor, op, create_elementwise(fn))

for op in unary_ops: setattr(EinTensor, op, (lambda name: lambda x, *args: EinTensor(x.einshape, getattr(x.data, name)(*args)))(op))

def create_reduce(fn, inverse = False):
  def wrapped (x, *axes:tuple[EinDim]):
    if len(axes) == 0: axes = x.einshape.dims
    if len(axes) == 1:
      if type(axes[0]) == EinTensor: axes = (axes[0].einshape,)
      if type(axes[0]) == einShape: axes = axes[0].dims
    data = x.data
    shp = x.einshape
    for k in [d for d in x.einshape.dims if d not in axes] if inverse else axes:
      data = fn(data, shp.dims.index(k))
      shp = shp.diff(k)
    res = EinTensor(shp, data)
    return res.reshape(*axes) if inverse else res
  return wrapped

for op in reduce_ops:
  setattr(EinTensor, op, create_reduce(getattr(Tensor, op)))
  setattr(EinTensor, f'{op}_to', create_reduce(getattr(Tensor, op), True))


if __name__ == '__main__':
  K,V,S,T = EinDim('K', 3), EinDim('V', 5), EinDim('S', 7), EinDim('T', 11)

  Dim, Nsamples, Out = EinDim('Dim', 10), EinDim('Nsamples', 100), EinDim('Out', 20)

  x = EinTensor.rand(Dim, Nsamples, Out)

  Ix = EinDim('Ix', 8)
  i = EinTensor([Ix], [1,2,3,4,5,6,7,8])
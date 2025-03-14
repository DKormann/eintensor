from tinygrad import Tensor
from tinygrad.ops import SimpleMathTrait
_dimensions = {}

class einDim:
  def __init__ (self,name:str, size:int):
    if name in _dimensions:
      assert _dimensions[name] == size, f"Dimension {name} already exists with size {_dimensions[name]}"
    _dimensions[name] = size
    self.name = name
    self.size = size
  
  def __repr__(self): return self.name
  def __eq__(self, other): return type(other) == einDim and  self.size == other.size and self.name == other.name
  def __hash__(self): return hash(self.name)

I = einDim('1', 1)

class einShape:

  def __init__(self, *dims):
    self.dims = dims
    self.shape = tuple(map(lambda x: x.size, dims))
  
  def __repr__(self):return f'einShape{self.dims}'

  def permute(self, order):
    assert len(order) == len(self.dims)
    return einShape(*[self.dims[i] for i in order])
  def broadcast (self, other): return einShape(*(list(dict.fromkeys(self.dims + other.dims))))
  def __eq__(self, other): return self.dims == other.dims

tensor_att = ['dtype', 'lazydata', 'numpy']
reduce_ops = ['sum', 'mean', 'max', 'min', 'prod', 'std', 'var']

class EinTensor():
  def __init__(self, shape:einShape, data:Tensor):
    assert shape.shape == data.shape
    self.einshape = shape
    self.data = data

  @staticmethod
  def ones(shape):
    if not type(shape) == einShape: shape = einShape(*shape)
    return EinTensor(shape, Tensor.ones(shape.shape))
  
  def zeros(shape): return EinTensor.ones(shape) * 0

  def __repr__(self): return f'einTensor({self.einshape}))'
  
  def permute(self, order): return EinTensor(self.einshape.permute(order), self.data.permute(order))

  def reshape (self, newshape): return EinTensor(newshape, self.data.reshape(newshape.shape))

  def expand(self, newshape):
    sparse_shape = self.einshape.dims + (I,) * (len(newshape.dims) - len(self.einshape.dims))
    reshaped = self.reshape(einShape(*sparse_shape))
    extra = len(self.einshape.dims)-1
    order = [self.einshape.dims.index(k) if k in self.einshape.dims else (extra := extra + 1) for k in newshape.dims]
    return reshaped.permute(order)
  
  def __getattribute__(self, name):
    if name in tensor_att:
      return getattr(self.data, name)
    return super().__getattribute__(name)

def create_elementwise(fn):
  def wrapped (one, other):
    if (type(other) == int or type(other) == float):
      val = other
      other = EinTensor.ones([])
      other.data *= val
    shape = one.einshape.broadcast(other.einshape)
    return EinTensor(
      one.einshape.broadcast(other.einshape),
      fn(one.expand(shape).data, other.expand(shape).data))
  return wrapped

binary_ops = [op for op in SimpleMathTrait.__dict__ if op not in EinTensor.__dict__]

for op in binary_ops:
  fn = getattr(SimpleMathTrait, op)
  setattr(EinTensor, op, create_elementwise(fn))

unary_ops = ['__neg__', 'abs', '__invert__', 'float', 'int', 'bool']
for op in unary_ops: setattr(EinTensor, op, (lambda name: lambda x: EinTensor(x.einshape, getattr(x.data, name)()))(op))

def create_reduce(fn, inverse = False):
  def wrapped (x, *axes:tuple[einDim]):
    if axes == []: axes = x.einshape.dims
    if type(axes) == einDim: axes = [axes]
    if inverse: axes = [d for d in x.einshape.dims if d not in axes]
    return EinTensor(einShape(*[k for k in x.einshape.dims if k not in axes]),
      fn(x.data, [x.einshape.dims.index(k) for k in axes]))
  return wrapped


for op in reduce_ops:
  setattr(EinTensor, op, create_reduce(getattr(Tensor, op)))
  setattr(EinTensor, f'{op}_to', create_reduce(getattr(Tensor, op), True))

if __name__ == '__main__':
  K,V,S,T = einDim('K', 3), einDim('V', 5), einDim('S', 7), einDim('T', 11)

  Dim, Nsamples, Out = einDim('Dim', 10), einDim('Nsamples', 100), einDim('Out', 20)

  x = EinTensor.ones((Nsamples, Dim))
  w = EinTensor.ones((Dim, Out))

  

  print((x * w).sum(Dim))
  print(x.einshape)
  print((x * w).sum_to(Nsamples, Out))
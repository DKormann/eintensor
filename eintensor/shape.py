from tinygrad import Tensor
from tinygrad.ops import SimpleMathTrait
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
    self.dims = dims
    self.shape = tuple(map(lambda x: x.size, dims))

  def __repr__(self):return f'einShape{self.dims}'

  def permute(self, order):
    assert len(order) == len(self.dims)
    return einShape(*[self.dims[i] for i in order])
  def broadcast (self, other): return einShape(*(list(dict.fromkeys(self.dims + other.dims))))
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

  def __repr__(self): return f'<einTensor [{', '.join(map(lambda x: x.name, self.einshape.dims))}]>'



  def expand(self, newshape):
    sparse_shape = einShape(*(self.einshape.dims + (I,) * (len(newshape.dims) - len(self.einshape.dims))))
    reshaped = EinTensor(sparse_shape, self.data.reshape(sparse_shape.shape))

    extra = len(self.einshape.dims)-1
    order = [self.einshape.dims.index(k) if k in self.einshape.dims else (extra := extra + 1) for k in newshape.dims]
    return EinTensor(reshaped.einshape.permute(order), reshaped.data.permute(order))
  
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

  EinDim, Nsamples, Out = EinDim('Dim', 10), EinDim('Nsamples', 100), EinDim('Out', 20)

  x = EinTensor.ones(Nsamples, EinDim)
  w = EinTensor.ones(EinDim, Out)

  print((x * w).sum(EinDim))
  print((x * w).sum_to(Nsamples, Out))
import unittest
from eintensor import EinTensor, einShape, EinDim

K,V,S,T = EinDim('K', 3), EinDim('V', 5), EinDim('S', 7), EinDim('T', 11)

x = EinTensor.ones(K,V)
y = EinTensor.ones(K,S,T)

class TestEinTensor(unittest.TestCase):

  def test_ones(self):
    x = EinTensor.ones(K,V)
    self.assertEqual(x.einshape, einShape(K,V), f'expected shape {einShape(K,V)} but got {x.einshape}')
  

  def test_binary(self):

    lams = [
      lambda a,b: a+b,
      lambda a,b: a*b,
      lambda a,b: a-b,
      lambda a,b: a/b,
    ]

    vals = [
      [x,x],
      [x,2],
      [3,x],
    ]

    for lam in lams:
      for [a,b] in vals:
        anump = getattr(a,'numpy', lambda : a)()
        bnump = getattr(b,'numpy', lambda: b)()
        assert (lam(a,b).numpy() == lam(anump, bnump)).all()
    

  def test_unary(self):
    lams = [
      lambda a: -a,
      lambda a: a.abs(),
      lambda a: ~a.bool(),
      lambda a: a.float(),
      lambda a: a.int(),
    ]
    for lam in lams: assert (lam(x).data == lam(x.data)).all().item()


  def test_getters(self):
    assert x.dtype == x.data.dtype

  def test_expand(self):

    assert x.expand(K,V,S,T).einshape == einShape(K,V,S,T), f'expected shape {einShape(K,V,S,T)} but got {x.expand(K,V,S,T).einshape}'
    self.assertEqual((x + y).einshape ,einShape(K,V,S,T))
    self.assertEqual((x - y).einshape ,einShape(K,V,S,T))
    self.assertEqual((x * y).einshape ,einShape(K,V,S,T))
  
  def test_permute(self):
    xp = x.permute(V,K)
    assert xp.einshape == einShape(V,K), f'expected shape {einShape(V,K)} but got {xp.einshape}'
    assert xp.data.shape == xp.einshape.shape, f'expected shape {xp.einshape.shape} but got {xp.data.shape}'
   
  def test_dtype(self):
    assert x.dtype == x.data.dtype
    assert (x.numpy() == x.data.numpy()).all()

  def test_stack(self):
    St = EinDim("S2", 2)
    s = x.stack(x, dim = St)
    assert s.einshape == einShape(St, K, V)

    s = x.stack(y, dim= St)
    self.assertEqual(s.einshape, einShape(St,K,V,S,T))

    assert((s.sum(St) == (x+y)).all().numpy())


  def test_getitem(self):
    assert x[None].einshape.dims == (EinDim(1), *x.einshape.dims)
    assert x[0].einshape.dims == x.einshape.dims[1:]
    self.assertEqual(x[:2].einshape.dims ,(EinDim(2), *x.einshape.dims[1:]))

  def test_tensor_attributes(self):
    x = EinTensor.rand(K,V, requires_grad = True)
    x.sum().backward()
    self.assertEqual(x.grad.einshape, x.einshape)
  
  def test_tensor_reduce(self):
    x = EinTensor.rand(K,V,S,T)
    assert x.sum().einshape == einShape()
    assert x.sum(K).einshape == einShape(V,S,T)
    assert x.sum_to(K).einshape == einShape(K)

    assert x.mean().einshape == einShape()
    assert x.mean_to(K).einshape == einShape(K)
    
    assert x.argmin(K).einshape == einShape(V,S,T)
    assert x.argmin_to(K,S).einshape == einShape(K,S)
    assert x.argmin_to(S,K).einshape == einShape(S,K)


if __name__ == '__main__':
  unittest.main()

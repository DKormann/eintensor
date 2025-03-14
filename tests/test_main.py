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
    self.assertEqual((x + y).einshape ,einShape(K,V,S,T))
    self.assertEqual((x - y).einshape ,einShape(K,V,S,T))
    self.assertEqual((x * y).einshape ,einShape(K,V,S,T))
  

  def test_dtype(self):
    assert x.dtype == x.data.dtype
    assert (x.numpy() == x.data.numpy()).all()
    


if __name__ == '__main__':
  unittest.main()

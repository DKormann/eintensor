import unittest
from tinsor import Dim, Shape

class TestShape(unittest.TestCase):
  def test_shape(self):
    
    shp = Shape(A=3, B=4)
    A, B = shp
    assert A.size == 3
    assert A.name == 'A'

    assert shp.keys == ['A', 'B']

    assert Shape(A, B) == shp
    assert Shape(A=3, B=4) == shp
    assert Shape(A=3, C=4) != shp
    assert Shape(A=3, B=5) != shp


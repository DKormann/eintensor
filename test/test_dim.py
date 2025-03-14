import unittest
from tinsor import Dim

class TestDim(unittest.TestCase):
  def test_dim(self):
    A = Dim('A', 3)

    assert A.size == 3
    assert A.name == 'A'

    A2 = Dim('A', 3)

    assert A2 == A

    B = Dim('B', 3)
    assert A != B
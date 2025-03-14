
## you like [tinygrad](https://github.com/tinygrad/tinygrad)? you know [einsum notation](https://rockt.github.io/2018/04/30/einsum)? than you could use tinsor

This tensor frontend will take care of all the unsqueese / permutes / expand you need


```python
from tinsor import Shape, Tensor

S, T, U, V = Shape(S=5, T=3, U=4, V=6)

x = Tensor.ones(S, T) # (S=5, T=3)
y = Tensor.rand(T, U) # (T=3, U=4)

p = x * y # (S=5, T=3, U=4)

print(p.numpy().shape) # (5, 3, 4)
```

### install

```bash
pip install -e .
```

formerly known as [tinsorchflow](https://www.youtube.com/watch?v=j5a0jTc9S10).



[code](https://github.com/dkormann/eintensor)

## idea
a sane way to handle many dimensions on Tensors
eintensor is using [tinygrad](github.com/tinygrad/tinygrad) under the hood



hello world:
```python
from eintensor import EinTensor, EinDim

FeatureDim, SeqLen, BatchSize, EmbeddingDim = EinDim('FeatureDim', 10), EinDim('SeqLen', 100), EinDim('BatchSize', 20), EinDim('EmbeddingDim', 30)

x = EinTensor.rand(SeqLen, BatchSize, FeatureDim)
w = EinTensor.rand(FeatureDim, EmbeddingDim)

pred = (x * w).sum_to(SeqLen, BatchSize, EmbeddingDim)

print(pred)         # <EinTensor [SeqLen, BatchSize, EmbeddingDim]>
print(pred.shape)   # (100, 20, 30)
```

Handling Tensor dimensions as numbered items in a shape, is akin to managing memory with pointers in C arrays.

Its possible but outdated.

Tensor dimensions should be named entities. You should think about the dimension on a high level and the compiler cares about ordering (reshaping, permuting, unsqueezing) the dimensions.

## EinDim

an EinDim is an object that represents the data dimension. it has a name and size

```python
K, V = EinDim('K', 10), EinDim('V', 20)
```

## EinTensor

#### create

an EinTensor is a tensor that has a einshape composed of EinDims

```python
O = EinTensor.ones(K, V)
R = EinTensor.rand(K, V)
```

### unary

unary operations work as expected from Tensor

### binary

you can perform binary operations on any two EinTensors.
If they share dimensions the operation is broadcasted on the shared dimensions.
Missing dimensions are expanded as needed.

The Developer does not have to remember the order of dimensions since it is never relevant to the result. 

```python
x = O + R  # shape: (K, V)

S = EinDim('S', 30)

# y will have expanded dimensions:
y = O + EinTensor.ones(K, S) # einshape: (K, V, S)
```

### reduce

reduce operations come in to flavors:

`x.sum(K)` will sum along the  K dimension

`x.sum_to(K,S)` will sum all other dimensions to get resulting dimension (K,) <reduce>_to will work for all reduce operations

supported reduce operations: [source](https://github.com/DKormann/eintensor/blob/a9e0501dde6f6792ed4c4d19bd45665de3bd4a2e/eintensor/tensor.py#L71)

### stack

you can stack any number of any tensors together you will get a new stack dimension or can provide a dimension. All tensors will be expanded to the same shape.

```
# this will create EinTensor of shape (StackDimension:2, K, V, S)

EinTensor.stack(
  EinTensor.ones(K,V),
  EinTensor.ones(V,S),
)
```


### indexing

supports fancy indexing like 

Dim, Nsamples, Out = EinDim('Dim', 10), EinDim('Nsamples', 100), EinDim('Out', 20)


Ix = EinDim('Ix', 8)
i = EinTensor([Ix], [1,2,3,4,5,6,7,8])

EinTensor.rand(Dim, Nsamples, Out) [i,i, 0]


[examples](https://github.com/dkormann/eintensor/tree/main/examples)
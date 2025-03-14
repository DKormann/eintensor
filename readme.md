
a sane way to handle many dimensions on Tensors

```python
from eintensor import EinTensor, EinDim

FeatureDim, SeqLen, BatchSize, EmbeddingDim = EinDim('FeatureDim', 10), EinDim('SeqLen', 100), EinDim('BatchSize', 20), EinDim('EmbeddingDim', 30)

x = EinTensor.rand(SeqLen, BatchSize, FeatureDim)
w = EinTensor.rand(FeatureDim, EmbeddingDim)

pred = (x * w).sum_to(SeqLen, BatchSize, EmbeddingDim)

print(pred)         # <EinTensor [SeqLen, BatchSize, EmbeddingDim]>
print(pred.shape)   # (100, 20, 30)
```

[read the docs](https://dkormann.com/eintensor)

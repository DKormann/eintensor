
a sane way to handle many dimensions on Tensors



```python
from eintensor import EinTensor, EinDim

FeatureDim, SeqLen, BatchSize, EmbeddingDim = EinDim('FeatureDim', 10), EinDim('SeqLen', 100), EinDim('BatchSize', 20), EinDim('EmbeddingDim', 30)

x = EinTensor.rand(SeqLen, BatchSize, FeatureDim)
w = EinTensor.rand(FeatureDim, EmbeddingDim)

pred = (x * w).sum_to(SeqLen, BatchSize, EmbeddingDim)

print(pred)
print(pred.shape)
```

#!/usr/bin/env python3
import numpy as np
import random

from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam
from extra.training import train, evaluate

from tinygrad import Tensor

class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

class Transformer:
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
    self.final = Tensor.scaled_uniform(embed_dim, syms)

  def forward(self, x):
    bs = x.shape[0]

    maxlen_eye = Tensor.eye(x.shape[1])
    maxlen_eye = maxlen_eye.unsqueeze(0).expand([bs, *maxlen_eye.shape])

    onehot_feat = x.one_hot(self.syms)

    onehot = maxlen_eye.cat(onehot_feat, dim=2).flatten(end_dim=1)

    x = onehot.dot(self.embed).reshape((bs, x.shape[1], -1))
    x = x.sequential(self.tbs)
    x = x.reshape((-1, x.shape[-1])).dot(self.final).log_softmax()
    return x.reshape((bs, -1, x.shape[-1]))



# dataset idea from https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py
def make_dataset():
  ds = []
  for i in range(100):
    for j in range(100):
      s = i+j
      ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
  random.shuffle(ds)
  ds = np.array(ds).astype(np.float32)
  ds_X = ds[:, 0:4]
  ds_Y = np.copy(ds[:, -4:])
  ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
  ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]
  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test

if __name__ == "__main__":
  model = Transformer(10, 4, 2, 128, 4, 32)
  X_train, Y_train, X_test, Y_test = make_dataset()
  lr = 0.003
  for i in range(10):
    optim = Adam(get_parameters(model), lr=lr)
    train(model, X_train, Y_train, optim, 50, BS=64, allow_jit=True)
    acc, Y_test_preds = evaluate(model, X_test, Y_test, num_classes=10, return_predict=True)
    lr /= 1.2
    print(f'reducing lr to {lr:.4f}')
  if acc > 0.998:
    wrong=0
    for k in range(len(Y_test_preds)):
      if (Y_test_preds[k] != Y_test[k]).any():
        wrong+=1
        a,b,c,x = X_test[k,:2].astype(np.int32), X_test[k,2:4].astype(np.int32), Y_test[k,-3:].astype(np.int32), Y_test_preds[k,-3:].astype(np.int32)
        print(f'{a[0]}{a[1]} + {b[0]}{b[1]} = {x[0]}{x[1]}{x[2]} (correct: {c[0]}{c[1]}{c[2]})')
    print(f'Wrong predictions: {wrong}, acc = {acc:.4f}')

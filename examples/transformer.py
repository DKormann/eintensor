#%%
from tinsor import Tensor, Shape, Dim
import random
import numpy as np
import matplotlib.pyplot as plt

#%%

def make_dataset():
  ds = []
  for i in range(100):
    for j in range(100):
      s = i+j
      ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
  random.shuffle(ds)
  ds = np.array(ds).astype(np.float32)
  ds_X = ds[:, 0:6]
  ds_Y = np.copy(ds[:, 1:])
  ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
  ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]
  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test

x_train, y_train, x_test, y_test = make_dataset()

# %%
plt.plot(x_train[4])
# %%


[1,2,3,4].__getitem__(slice(0,4,2))


# %%

import dataclasses


@dataclasses.dataclass
class Idx:
  i :int

  def __getitem__(self, idx):
    print(idx)

I = Idx(1)


I[0: I]
  
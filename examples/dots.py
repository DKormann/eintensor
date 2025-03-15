from PIL import Image
from eintensor import EinTensor, EinDim
import random

Width, Height = EinDim('Width', 100), EinDim('Height', 100)
RGB = EinDim('RGB', 3)

X = EinTensor.linspace(-1, 1, Height)
Y = X.reshape(Width)

Ndots = EinDim('Ndots', 10)
PosDim = EinDim("Pos:2", 2)
RandPos = EinTensor.rand(Ndots, PosDim) * 2 - 1
Pos = X.stack(Y, dim = PosDim)
XY = (Pos + RandPos)/0.2

dots = (1-(XY ** 2).sum(PosDim).sqrt()).clamp(0,1)
colors = EinTensor.randint(Ndots, RGB, high = 255)
x = (dots * colors).sum(Ndots).clamp(0,255) 

print(x.data)

arr = x.permute(Width, Height, RGB).numpy().astype('uint8')
img = Image.fromarray(arr, 'RGB')

img.save(__file__[:-3]+'.jpg')


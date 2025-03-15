from PIL import Image
from eintensor import EinTensor, EinDim

Width, Height = EinDim('Width', 100), EinDim('Height', 100)

RGB = EinDim('RGB', 3)

X = EinTensor.linspace(0, 1, Height)
Y = X.reshape(Width)

XY = X.stack(Y)
XY = XY * 2 - 1 # 0 at the center

dist = (XY ** 2).sum_to(Width, Height).sqrt()
dist = 1-dist.clamp(0,1)

color = EinTensor([RGB], [0,255,255])

x = dist * color

arr = x.permute(Width, Height, RGB).numpy().astype('uint8')
img = Image.fromarray(arr, 'RGB')
img.save('./img.jpg')


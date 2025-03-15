from PIL import Image
from eintensor import EinTensor, EinDim

Width, Height = EinDim('Width', 100), EinDim('Height', 100)
RGB = EinDim('RGB', 3)

X = EinTensor.linspace(-1, 1, Height)
Y = X.reshape(Width)

XY = X.stack(Y)
dist =  (XY ** 2).sum_to(Width, Height).sqrt()

color = EinTensor([RGB], [0,255,255])
x = color * (1-dist.clamp(0,1))

arr = x.permute(Width, Height, RGB).numpy().astype('uint8')
img = Image.fromarray(arr, 'RGB')

img.save(__file__[:-3]+'.jpg')


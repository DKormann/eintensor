from PIL import Image

from eintensor import EinTensor, EinDim



Width, Height = EinDim('Width', 100), EinDim('Height', 100)


R = EinTensor.linspace(0, 255, Width)
G = EinTensor.linspace(0, 255, Height)

x = EinTensor.stack(R, G, G*0, dim= (RGB:=EinDim('RGB', 3)))

arr = x.permute(Width, Height, RGB).numpy().astype('uint8')
img = Image.fromarray(arr, 'RGB')
img.save('./img.jpg')


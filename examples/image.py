from PIL import Image

from eintensor import EinTensor, EinDim



Width, Height = EinDim('Width', 100), EinDim('Height', 100)


x = EinTensor.zeros(Width, Height,3)
y = EinTensor.linspace(0, 255, Width)


x = x + y


arr = x.numpy().astype('uint8')
print(arr[0,0], arr[-1,-1])

img = Image.fromarray(arr, 'RGB')


img.save('./img.jpg')


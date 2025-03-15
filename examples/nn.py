from eintensor import EinTensor, EinDim

Nsamples = EinDim("Nsamples", 20)
Feature = EinDim("Feature", 50)
Out = EinDim("Out", 10)

x = EinTensor.rand(Nsamples, Feature)

w = EinTensor.rand(Feature, Out, requires_grad = True)

p = x @ w
loss = p.sum()

print(loss.backward().numpy())

print(w.std().numpy())
print(w.grad.std().numpy())

w = w - (w.grad * 0.000001)

p = x @ w
loss = p.sum()

print(loss.numpy())

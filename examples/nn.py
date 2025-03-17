from eintensor import EinTensor, EinDim

Nsamples = EinDim("Nsamples", 20)
Feature = EinDim("Feature", 50)
Out = EinDim("Out", 10)

x = EinTensor.rand(Feature)

w = EinTensor.rand(Feature, Out, requires_grad = True)



p = x @ w
loss = p.sum()
loss.backward()
# print(loss.numpy())


ass


print((w**2).mean().numpy(), (w.grad**2).mean().numpy())

print((w.grad < w).numpy().shape)




p = x @ w
loss = p.sum()

# print(loss.numpy())



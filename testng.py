from noisygrad import NoisyGrad
import torch

ng = NoisyGrad(.1)
inputv = torch.tensor([1.,2.,3.]*10, requires_grad=True)
values = ng.apply(inputv)
print(f"{values=}")

sum_value = torch.sum(values)
sum_value.backward()

print(f"{sum_value=}")
print(f"{inputv.grad=}")
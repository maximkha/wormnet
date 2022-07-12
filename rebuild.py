from wormnet import WormNet, worm_classification_forward
import torch
import torch.nn as nn
import torch.optim as optim

connectome = torch.tensor([[ 4.6771,  4.6771,  0.0000,  0.0000,  0.0000],
        [ 4.6771,  4.6771,  0.0000,  0.0000,  0.0000],
        [ 3.1797,  3.1797,  0.0000,  0.0000,  0.0000],
        [-3.6333, -3.6333,  7.6649,  4.7073,  1.8020],
        [ 3.9005,  3.9005, -1.2367, -3.7195,  3.1055]], requires_grad=False)
print(f"{connectome=}")
clock_neuron = torch.tensor([0.0804, 0.0804, 7.6967, 4.3912, 1.5926], requires_grad=False)
print(f"{clock_neuron=}")

wn = WormNet(5, 7.)
wn.connectome = nn.parameter.Parameter(connectome)
wn.clock_neuron = nn.parameter.Parameter(clock_neuron)

Xs = torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,1.]])
# outp = worm_classification_forward(wn, Xs, 2, 2, None)
outp = worm_classification_forward(wn, Xs, 2, 2, None)
print(f"{outp=}")
print(f"{torch.softmax(outp/.01, axis=1)=}")


print(f"+++++ROUNDING+++++")

wn.connectome = nn.parameter.Parameter(torch.round(connectome))
wn.clock_neuron = nn.parameter.Parameter(torch.round(clock_neuron))

outp = worm_classification_forward(wn, Xs, 2, 2, None)
print(f"{outp=}")
print(f"{torch.softmax(outp/.01, axis=1)=}")
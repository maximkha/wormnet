from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch

class SimpleStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0., 1., 0.)
    @staticmethod
    def backward(ctx, grad_outp):
        x, = ctx.saved_tensors
        sigmoid_at_x = 1./(1. + torch.exp(-x))
        return grad_outp * sigmoid_at_x * (1. -sigmoid_at_x)
        # return grad_outp
def apply_ss(x):
    return SimpleStep.apply(x)

def sim_step(state: torch.Tensor, fired: torch.Tensor, connectome: torch.Tensor, clock_neuron: torch.Tensor, thresh: float):
    state += F.linear(fired, connectome, clock_neuron)
    fired = apply_ss(state -thresh)

    new_state = state * (1-fired)
    return new_state, fired


class DiscreteWormNet(nn.Module):
    def __init__(self, nneuron, thresh = 0.):
        super(DiscreteWormNet, self).__init__()
        self.thresh = thresh
        # self.connectome = Parameter(torch.zeros((nneuron, nneuron)))
        self.connectome = Parameter(torch.normal(0, 1, (nneuron, nneuron)))
        # self.clock_neuron = Parameter(torch.zeros((nneuron,)))
        self.clock_neuron = Parameter(torch.normal(0, 1, (nneuron,)))
    
    def forward(self, state: torch.Tensor, fired: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return sim_step(state, fired, self.connectome, self.clock_neuron, self.thresh)

def worm_classification_forward(worm: DiscreteWormNet, inputs, n_out, n_step: int):
    fired = torch.zeros((inputs.shape[0], worm.clock_neuron.shape[0])).to(inputs.device)
    state = torch.zeros((inputs.shape[0], worm.clock_neuron.shape[0])).to(inputs.device)
    fired[:, :inputs.shape[1]] = inputs
    current_out = torch.zeros(inputs.shape[0], n_out).to(inputs.device)
    for _ in range(n_step):
        state, fired = worm(state, fired)
        current_out += fired[:, -n_out:]
    
    return current_out
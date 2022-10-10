from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from noisygrad import NoisyGrad

# nneuron = 10
# thresh = 8.
# connectome = torch.zeros((nneuron, nneuron))
# clock_neuron = torch.zeros((nneuron,))

# xs = torch.tensor([[0,0], [0,1], [1,0], [1,1]])
# ys = torch.tensor([[0],[0],[0],[1]]) + 2

# state = torch.zeros((nneuron))

# fired = torch.zeros_like(state)
# fired[0] = 1.
# fired[1] = 1.

def sim_fuzzy_step(state: torch.Tensor, fired: torch.Tensor, connectome: torch.Tensor, clock_neuron: torch.Tensor, thresh: float, inv_temp: float):
    # tempered update
    state += F.linear(fired, connectome, clock_neuron)
    fired = torch.sigmoid(inv_temp*(state -thresh)) # mark as fired
    new_state = state * (1-fired)
    return new_state, fired

def sim_perfect_step(state, fired, connectome, clock_neuron, thresh):
    # perfect update
    state += F.linear(fired, connectome, clock_neuron)
    fired = torch.zeros_like(state)
    fired[state > thresh] = 1. # mark as fired
    state[state > thresh] = 0. # reset fired neurons
    return state, fired
    # diffs = state -thresh
    # fired = diffs / torch.abs(diffs)
    # new_state = state * (1-fired)
    # return new_state, fired

class WormNet(nn.Module):
    def __init__(self, nneuron, thresh = 8.):
        super(WormNet, self).__init__()
        self.thresh = thresh
        self.connectome = Parameter(torch.zeros((nneuron, nneuron)))
        self.clock_neuron = Parameter(torch.zeros((nneuron,)))
    
    def forward(self, inv_temp: float, state: torch.Tensor, fired: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return sim_fuzzy_step(state, fired, self.connectome, self.clock_neuron, self.thresh, inv_temp)

    def perfect_forward(self, state: torch.Tensor, fired: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return sim_perfect_step(state, fired, self.connectome, self.clock_neuron, self.thresh)

# first neurons are input
# last n_out are output
def worm_classification_forward(worm: WormNet, inputs, n_out, n_step, inv_temp):
    fired = torch.zeros((inputs.shape[0], worm.clock_neuron.shape[0])).to(inputs.device)
    state = torch.zeros((inputs.shape[0], worm.clock_neuron.shape[0])).to(inputs.device)
    fired[:, :inputs.shape[1]] = inputs
    current_out = torch.zeros(inputs.shape[0], n_out).to(inputs.device)
    for _ in range(n_step):
        if inv_temp == None:
            state, fired = worm.perfect_forward(state, fired)
        else:
            state, fired = worm(inv_temp, state, fired)
        current_out += fired[:, -n_out:]
    
    return current_out

# max first fire
def mff_loss(fire_snapshots, true_fire, false_fire):
    return torch.mean(torch.cat((torch.max(fire_snapshots[:, true_fire], 0).values -1, torch.max(fire_snapshots[:, false_fire], 0).values))**2)

def bff_loss(fire_snapshots, true_fire, false_fire):
    raise NotImplementedError()
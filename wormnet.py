from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

def sim_fuzzy_step(state, fired, connectome, clock_neuron, thresh, inv_temp):
    # tempered update
    state += F.linear(fired, connectome, clock_neuron)
    fired = torch.sigmoid(inv_temp*(state -thresh)) # mark as fired
    state -= state * fired # reset fired neurons
    return state, fired

def sim_perfect_step(state, fired, connectome, clock_neuron, thresh):
    # perfect update
    state += F.linear(fired, connectome, clock_neuron)
    fired = torch.zeros_like(state)
    fired[state > thresh] = 1. # mark as fired
    state[state > thresh] = 0. # reset fired neurons
    return state, fired

class WormNet(nn.Module):
    def __init__(self, nneuron, thresh = 8.):
        self.thresh = thresh
        self.connectome = Parameter(torch.zeros((nneuron, nneuron)))
        self.clock_neuron = Parameter(torch.zeros((nneuron,)))

        super().__init__()
    
    def forward(self, inv_temp, state: torch.Tensor, fired: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return sim_fuzzy_step(state, fired, self.connectome, self.clock_neuron, self.thresh, inv_temp)

# max first fire
def mff_loss(fire_snapshots, true_fire, false_fire):
    return torch.mean(torch.cat((torch.max(fire_snapshots[:, true_fire], 0).values -1, torch.max(fire_snapshots[:, false_fire], 0).values))**2)

def bff_loss(fire_snapshots, true_fire, false_fire):
    raise NotImplementedError()
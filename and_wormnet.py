import wormnet
import torch

nneuron = 5
thresh = 8.
connectome = torch.zeros((nneuron, nneuron))
clock_neuron = torch.zeros((nneuron,))
state = torch.zeros((nneuron))
fired = torch.zeros_like(state)
fired[0] = 1.
fired[1] = 1.

# connectome[2, 0] = 5.
# connectome[2, 1] = 5.
# connectome[4, 3] = 10.
# clock_neuron[3] = 10.

connectome[3, 0] = 5.
connectome[3, 1] = 5.
connectome[4, 2] = 10.
clock_neuron[2] = 10.

print(f"{clock_neuron}")

for i in range(5):
    state, fired = wormnet.sim_fuzzy_step(state, fired, connectome, clock_neuron, thresh, 2)
    print(f"STEP: {i}")
    print(f"{state=}")
    print(f"{fired=}")
    print(f"fire_1 = {fired[3]}, fire_0 = {fired[4]}")
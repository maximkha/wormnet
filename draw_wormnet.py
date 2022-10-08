import networkx as nx
from collections import Counter
import numpy as np

# # g = nx.DiGraph((x, y, {'weight': v}) for (x, y), v in Counter(EDGES).items())
connectome = np.array([[ 4.6771,  4.6771,  0.0000,  0.0000,  0.0000],
                       [ 4.6771,  4.6771,  0.0000,  0.0000,  0.0000],
                       [ 3.1797,  3.1797,  0.0000,  0.0000,  0.0000],
                       [-3.6333, -3.6333,  7.6649,  4.7073,  1.8020],
                       [ 3.9005,  3.9005, -1.2367, -3.7195,  3.1055]])
print(f"{connectome=}")
connectome = np.round(connectome)
clock_neuron = np.array([0.0804, 0.0804, 7.6967, 4.3912, 1.5926])
clock_neuron = np.round(clock_neuron)

# connectome = np.zeros((5, 5))
# clock_neuron = np.zeros((5,))

# connectome[3, 0] = 5.
# connectome[3, 1] = 5.
# connectome[4, 3] = -5.
# connectome[4, 2] = 10.
# clock_neuron[2] = 10.

EDGES = []
for i in range(connectome.shape[0]):
    for j in range(connectome.shape[0]):
        weight = connectome[i, j]
        if weight != 0:
            print(f"connected neuron_{j} --{weight}--> neuron_{i}")
            EDGES.append((f"neuron_{j}", f"neuron_{i}", weight))

for i in range(connectome.shape[0]):
    weight = clock_neuron[i]
    if weight != 0:
        print(f"clock --{weight}--> neuron_{i}")
        EDGES.append((f"clock", f"neuron_{i}", weight))

g = nx.DiGraph((x, y, {'weight': v}) for x, y, v in EDGES)

NEURON_LIST = [f"neuron_{i}" for i in range(connectome.shape[0])] + ["clock"]

import matplotlib.pyplot as plt; plt.close('all')
from matplotlib.animation import FuncAnimation

def animate_nodes(G, node_colors, pos=None, *args, **kwargs):

    # define graph layout if None given
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=20)

    # draw graph
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=NEURON_LIST, node_color=np.zeros(len(G.nodes)), cmap="ocean", vmin=-11, vmax=15)
    edges = nx.draw_networkx_edges(G, pos, *args, **kwargs)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.set_cmap('bwr')
    # plt.colorbar()
    # plt.clim(-10,10)
    # plt.axis('off')

    def update(ii):
        # nodes are just markers returned by plt.scatter;
        # node color can hence be changed in the same way like marker colors
        nodes.set_array(node_colors[ii])
        return nodes,

    fig = plt.gcf()
    animation = FuncAnimation(fig, update, interval=50, frames=len(node_colors), blit=True)
    return animation

total_nodes = len(g.nodes)
time_steps = 20
# node_colors = np.random.randint(0, 100, size=(time_steps, total_nodes))

def smooth_between(a, b, t, k=10):
    return a + ((b-a)/(1 + np.exp(-k * (t-.5))))

def smooth_peak(t, k=5):
    return 1/(1 + np.exp(-np.sin(2*np.pi*(t-.25))*k))

raw_node_colors = []
# do simulation and create node colors
step_frames = 100

print(f"{list(g)=}")

from wormnet import sim_perfect_step
import torch
fired = torch.zeros(total_nodes-1).float()
state = torch.zeros(total_nodes-1).float()
fired[:2] = torch.tensor([1.,1.])
frames = []

first_color_frame = np.zeros((step_frames*2, total_nodes,))
for i, has_fired in enumerate(list(fired)):
    if has_fired.item() == 1:
        first_color_frame[step_frames:int(step_frames + step_frames/2), i] = smooth_between(state[i].item(), 10, np.linspace(0, 1, int(step_frames/2)))
        first_color_frame[int(step_frames + step_frames/2):, i] = smooth_between(10, 0, np.linspace(0, 1, int(step_frames/2)))
first_color_frame[step_frames:, -1] = 10*smooth_peak(np.linspace(0, 1, step_frames))
frames.append(first_color_frame)


for _ in range(2):
    color_frame = np.zeros((step_frames*2, total_nodes,))
    color_frame[step_frames:, -1] = 10*smooth_peak(np.linspace(0, 1, step_frames))
    old_state = state.clone()
    state, fired = sim_perfect_step(state, fired, torch.tensor(connectome).float(), torch.tensor(clock_neuron).float(), 7.)
    for i in range(total_nodes-1):
        color_frame[:step_frames, i] = old_state[i].item()
        if fired[i].item() != 1:
            # print(f"{old_state[i]=}")
            # print(f"{state[i]=}")
            # print(f"{color_frame[:step_frames, i]=}")
            color_frame[:step_frames, i] = smooth_between(old_state[i].item(), state[i].item(), np.linspace(0, 1, step_frames))
            color_frame[step_frames:, i] = state[i].item()

    for i, has_fired in enumerate(list(fired)):
        if has_fired.item() == 1:
            color_frame[step_frames:int(step_frames + step_frames/2), i] = smooth_between(state[i].item(), 10, np.linspace(0, 1, int(step_frames/2)))
            color_frame[int(step_frames + step_frames/2):, i] = smooth_between(10, 0, np.linspace(0, 1, int(step_frames/2)))
    frames.append(color_frame)
    # color_frame.append()
# print(f"{color_frame=}")
# exit()
# animation = animate_nodes(g, node_colors)
animation = animate_nodes(g, np.vstack(frames))
animation.save('test.gif', writer='Pillow', savefig_kwargs={'facecolor':'white'}, fps=60)
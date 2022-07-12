import numpy as np

connectome = np.array([[ 4.6771,  4.6771,  0.0000,  0.0000,  0.0000],
        [ 4.6771,  4.6771,  0.0000,  0.0000,  0.0000],
        [ 3.1797,  3.1797,  0.0000,  0.0000,  0.0000],
        [-3.6333, -3.6333,  7.6649,  4.7073,  1.8020],
        [ 3.9005,  3.9005, -1.2367, -3.7195,  3.1055]])
print(f"{connectome=}")

clock_neuron = np.array([0.0804, 0.0804, 7.6967, 4.3912, 1.5926])
print(f"{clock_neuron=}")

abscon = np.abs(connectome)
absclock = np.abs(clock_neuron)

new_val = abscon.T @ np.ones_like(absclock) + absclock
print(f"{new_val / new_val.max() =}")
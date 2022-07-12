import numpy as np
import torch

def clipped_sigmoid_appx(x:np.ndarray, k: float=-.01) -> np.ndarray:
    out = np.where((x >= 0.) & (x <= 1.), x, -k*x)
    out += np.where(x > 1., 1. + k, 0.)
    return out

def clipped_sigmoid_appx_torch(x: torch.Tensor, k: float=-.01) -> torch.Tensor:
    out = torch.where((x >= 0.) & (x <= 1.), x, -k*x)
    out += torch.where(x > 1., 1. + k, 0.)
    return out

def clamp_appx(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x+1.)/2., 0, 1)
    # return torch.clamp(x, 0, 1)

# import matplotlib.pyplot as plt

# Xs = np.linspace(-2, 2, num=101)
# plt.plot(Xs, clipped_sigmoid_appx(Xs))
# plt.show()
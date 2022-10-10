import torch

def NoisyGrad(standard_deviation):
    class InnerNoisyGrad(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            return x.clone()

        @staticmethod
        def backward(self, grad_out):
            return grad_out + standard_deviation*torch.randn(*grad_out.shape)

    return InnerNoisyGrad
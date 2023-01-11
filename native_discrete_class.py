from native_discrete import DiscreteWormNet, worm_classification_forward
import torch
import torch.nn as nn
import torch.optim as optim

N_INPUT_NEURON = 2
N_OUTPUT_NEURON = 2
N_INTERMEDIATE_NEURON = 5
N_PROP_STEP = 4
THRESH = 1.

N_NEURON = N_INPUT_NEURON + N_OUTPUT_NEURON + N_INTERMEDIATE_NEURON

wormn = DiscreteWormNet(N_NEURON, THRESH)

# wormn = torch.jit.script(wormn)

Xs = torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,1.]])

# classes = torch.tensor([0,0,0,1])
# classes = torch.tensor([0,0,1,0])
# classes = torch.tensor([0,1,1,0])
# classes = torch.tensor([1,1,0,0])
classes = torch.tensor([1,1,1,0])

torch.autograd.set_detect_anomaly(True)

opt = optim.Adam(wormn.parameters(), lr=1e-3)
loss_func = nn.CrossEntropyLoss()

# def l2_reg_pen(model: nn.Module):
#     l2_total = 0.

#     for param in model.parameters():
#         l2_total += torch.sum(param.abs())

#     return l2_total

MAX_EPOCH = 10000
for i in range(MAX_EPOCH):
    opt.zero_grad()
    outp = worm_classification_forward(wormn, Xs, N_OUTPUT_NEURON, N_PROP_STEP)
    cat_outp = torch.softmax(outp, axis=1)

    loss = loss_func.forward(cat_outp, classes)
    loss.backward(create_graph = False)
    opt.step()

    if i % 200 == 199:
        print(f"loss={loss}")
        print(f"epoch={i}")
        print(f"cat_outp={cat_outp}")

    if loss.item() < 0.35:
        print(f"OK loss={loss}")
        print(f"cat_outp={cat_outp}")
        break


print(f"\nloss={loss}")

print("++++++++++++TEST++++++++++++")
outp = worm_classification_forward(wormn, Xs, N_OUTPUT_NEURON, N_PROP_STEP)
cat_outp = torch.softmax(outp, axis=1)
print(f"outp={outp}")
print(f"outp={torch.softmax(outp/.01, axis=1)}")

print("++++++++++++CONNECTOME AND CLOCK++++++++++++")
print(f"{wormn.connectome=}")
print(f"{wormn.clock_neuron=}")
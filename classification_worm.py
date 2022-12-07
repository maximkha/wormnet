from wormnet import WormNet, worm_classification_forward
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch_optimizer as moreoptim

N_INPUT_NEURON = 2
N_OUTPUT_NEURON = 2
N_INTERMEDIATE_NEURON = 0
N_PROP_STEP = 4
THRESH = 1.
TRAIN_TEMP = .5

N_NEURON = N_INPUT_NEURON + N_OUTPUT_NEURON + N_INTERMEDIATE_NEURON

wormn = WormNet(N_NEURON, THRESH)

Xs = torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,1.]])
# Ys = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])

classes = torch.tensor([0,0,0,1])
# classes = torch.tensor([0,0,1,0])
# classes = torch.tensor([0,1,1,0])
# classes = torch.tensor([1,1,0,0])
# classes = torch.tensor([1,1,1,0])

torch.autograd.set_detect_anomaly(True)

# opt = moreoptim.Yogi(wormn.parameters(), lr=2e-1)
# opt = moreoptim.Lookahead(optim.Adam(wormn.parameters(), lr=2e-1))
# opt = optim.Adam(wormn.parameters(), lr=1e-1)
# opt = optim.Adam(wormn.parameters(), lr=1e-1)
# opt = optim.Adam(wormn.parameters(), lr=5e-1)
opt = optim.Adam(wormn.parameters(), lr=1e-1)
# opt = optim.AdamW(wormn.parameters(), lr=7.5e-3)
# opt = optim.AdamW(wormn.parameters(), lr=3e-1)
# opt = optim.AdamW(wormn.parameters(), lr=1e-2)
loss_func = nn.CrossEntropyLoss()

def l2_reg_pen(model: nn.Module):
    l2_total = 0.

    for param in model.parameters():
        l2_total += torch.sum(param.abs())

    return l2_total

MAX_EPOCH = 10000
for i in range(MAX_EPOCH):
    opt.zero_grad()
    outp = worm_classification_forward(wormn, Xs, N_OUTPUT_NEURON, N_PROP_STEP, TRAIN_TEMP)
    cat_outp = torch.softmax(outp, axis=1)

    loss = loss_func.forward(cat_outp, classes)
    # loss = loss_func.forward(cat_outp, classes) + l2_reg_pen(wormn) * .001
    loss.backward(create_graph = False)
    opt.step()

    if i % 200 == 199:
        print(f"loss={loss}")
        print(f"epoch={i}")
        print(f"cat_outp={cat_outp}")

    if loss.item() < 0.3173:
        print(f"OK loss={loss}")
        print(f"cat_outp={cat_outp}")
        break

MAX_EPOCH = 2000
for invtemp in np.linspace(1,10000,16):
    print(f"{invtemp=}")
    for i in range(MAX_EPOCH):
        opt.zero_grad()
        outp = worm_classification_forward(wormn, Xs, N_OUTPUT_NEURON, N_PROP_STEP, invtemp)/.1
        cat_outp = torch.softmax(outp, axis=1)

        loss = loss_func.forward(cat_outp, classes)
        loss.backward()
        opt.step()
        # print(f"loss={loss}")
        if i % 200 == 199:
            print(f"loss={loss}")
            print(f"epoch={i}")
            print(f"cat_outp={cat_outp}")
        
        if loss.item() < 0.3173:
            print(f"OK loss={loss}")
            print(f"cat_outp={cat_outp}")
            break


print(f"\nloss={loss}")

print("++++++++++++TEST++++++++++++")
outp = worm_classification_forward(wormn, Xs, 2, 3, 1)/.1
cat_outp = torch.softmax(outp, axis=1)
print(f"outp={outp}")
print(f"outp={torch.softmax(outp/.01, axis=1)}")

print("++++++++++++COLDER++++++++++++")
outp = worm_classification_forward(wormn, Xs, 2, 3, 2)/.1
cat_outp = torch.softmax(outp, axis=1)
print(f"outp={outp}")
print(f"outp={torch.softmax(outp/.01, axis=1)}")

print("++++++++++++CONNECTOME AND CLOCK++++++++++++")
print(f"{wormn.connectome=}")
print(f"{wormn.clock_neuron=}")

print("++++++++++++ 0 ++++++++++++")
wormn.thresh = THRESH
outp = worm_classification_forward(wormn, Xs, N_OUTPUT_NEURON, N_PROP_STEP, None)/.1
print(f"{outp=}")
print(f"{torch.softmax(outp/.01, axis=1)=}")
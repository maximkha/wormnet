from wormnet import WormNet, worm_classification_forward
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

wormn = WormNet(15, 8.5)
# wormn_jit = torch.jit.script(wormn)
# connectome = torch.zeros((5, 5))
# clock_neuron = torch.zeros((5,))

# connectome[3, 0] = 5.
# connectome[3, 1] = 5.
# connectome[4, 3] = -5.
# connectome[4, 2] = 10.
# clock_neuron[2] = 10.

# wormn.connectome = torch.nn.parameter.Parameter(connectome)
# wormn.clock_neuron = torch.nn.parameter.Parameter(clock_neuron)

# outp = sim(wormn, torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,1.]]), 2, 2, 3)
# print(f"outp={outp}")
# print(f"outp={torch.softmax(outp/.01, axis=1)}")

Xs = torch.tensor([[1.,0.],[0.,1.],[0.,0.],[1.,1.]])
# Ys = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
# classes = torch.tensor([0,0,0,1])
classes = torch.tensor([0,0,1,0])
# classes = torch.tensor([0,1,1,0])
# classes = torch.tensor([1,1,0,0])

torch.autograd.set_detect_anomaly(True)

opt = optim.AdamW(wormn.parameters(), lr=1e-1)
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
    outp = worm_classification_forward(wormn, Xs, 2, 7, .5)
    cat_outp = torch.softmax(outp, axis=1)

    loss = loss_func.forward(cat_outp, classes)
    # loss = loss_func.forward(cat_outp, classes) + l2_reg_pen(wormn) * .001
    loss.backward()
    opt.step()
    # print(f"loss={loss}")
    if i % 200 == 199:
        print(f"loss={loss}")
        print(f"epoch={i}")
        print(f"cat_outp={cat_outp}")

MAX_EPOCH = 2000
for invtemp in np.linspace(1,5,16):
    print(f"{invtemp=}")
    for i in range(MAX_EPOCH):
        opt.zero_grad()
        outp = worm_classification_forward(wormn, Xs, 2, 7, 2)/.1
        cat_outp = torch.softmax(outp, axis=1)

        loss = loss_func.forward(cat_outp, classes)
        loss.backward()
        opt.step()
        # print(f"loss={loss}")
        if i % 200 == 199:
            print(f"loss={loss}")
            print(f"epoch={i}")
            print(f"cat_outp={cat_outp}")
        
        if loss < 0.3173: break


print(f"\nloss={loss}")

print("++++++++++++TEST++++++++++++")
outp = worm_classification_forward(wormn, Xs, 2, 7, 1)/.1
cat_outp = torch.softmax(outp, axis=1)
print(f"outp={outp}")
print(f"outp={torch.softmax(outp/.01, axis=1)}")

print("++++++++++++COLDER++++++++++++")
outp = worm_classification_forward(wormn, Xs, 2, 7, 2)/.1
cat_outp = torch.softmax(outp, axis=1)
print(f"outp={outp}")
print(f"outp={torch.softmax(outp/.01, axis=1)}")

print("++++++++++++CONNECTOME AND CLOCK++++++++++++")
print(f"{wormn.connectome=}")
print(f"{wormn.clock_neuron=}")

print("++++++++++++ 0 ++++++++++++")
wormn.thresh = 8.5
outp = worm_classification_forward(wormn, Xs, 2, 7, None)/.1
print(f"{outp=}")
print(f"{torch.softmax(outp/.01, axis=1)=}")
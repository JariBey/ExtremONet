import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch
import random

class ExtremeLearning(nn.Module):
    def __init__(self, indim, outdim, c=1, s=1, acfunc=nn.Tanh(), norm=[0, 1], device=None):
        super(ExtremeLearning, self).__init__()
        self.outdim = outdim
        self.indim = indim
        self.af = acfunc
        self.norm = norm
        self.c = c
        self.s = s
        self.device = device
        self.R = self.init_R(indim,outdim,c)
        self.b = torch.rand(1, outdim).to(self.device)*2-1

    def init_R(self,indim,outdim,c):
        res = torch.rand(indim, outdim) * 2 - 1
        res /= torch.sqrt(torch.tensor(indim * c, dtype=torch.float32))
        p = torch.rand(res.shape) > self.c
        res[p] = 0
        return res.to(self.device)
    
    def init_R(self,indim,outdim,c):
        res = torch.zeros(outdim,indim)
        for i in range(outdim):
            if c<indim:
                randinds = np.random.choice(range(indim),c,replace=False)
                res[i,randinds] =torch.rand(c)/np.sqrt(c)*2-1
            else:
                randinds = np.arange(indim)
                res[i,randinds] = torch.rand(indim)/np.sqrt(indim)*2-1
        return res.T.to(self.device)

    def scale(self, x):
        return (x - self.norm[0]) / self.norm[1]

    def forward(self, x):
        x = x.to(self.device)
        y = self.scale(x)
        y = torch.matmul(y, self.R.to(self.device)) + self.b.to(self.device)
        return self.af(self.s * y)


class HierarchicalExtremeLearning(nn.Module):
    def __init__(self, indim, outdim,layers, c=1, s=1, acfunc=nn.Tanh(), norm=[0, 1], device=None):
        super(HierarchicalExtremeLearning, self).__init__()
        self.outdim = outdim
        self.layers = layers
        self.indim = indim
        self.af = acfunc
        self.norm = norm
        self.c = c
        self.s = s
        self.device=device
        self.Rs = [self.init_R(indim,outdim,c[0])]+[self.init_R(outdim,outdim,c[i+1]) for i in range(layers-1)]
        self.bs = [torch.randn(1, outdim).to(self.device) for i in range(layers)]
    def init_R(self,indim,outdim,c):
        res = torch.rand(indim, outdim) * 2 - 1
        res /= torch.sqrt(torch.tensor(indim*c, dtype=torch.float32))
        p = torch.rand(res.shape) > c
        res[p] = 0
        return res.to(self.device)
    def scale(self, x):
        return (x - self.norm[0]) / self.norm[1]
    def forward(self, x):
        x = x.to(self.device)
        y = self.scale(x)
        ys = []
        for i in range(self.layers):
            y = self.af(self.s[i]*(torch.matmul(y, self.Rs[i]) + self.bs[i]))
            ys.append(y)
        return torch.cat(ys, dim=1)

class ExtremONet(nn.Module):
    def __init__(self, outdim, psize, Trunknet, Branchnet, loss_func=None, device=None):
        super(ExtremONet, self).__init__()
        self.outdim = outdim
        self.psize = psize
        self.trunk = Trunknet
        self.branch = Branchnet
        self.device=device
        self.B = nn.Parameter(torch.zeros(1, outdim), requires_grad=False)
        self.A = nn.Parameter(torch.zeros(psize, outdim), requires_grad=False)
        self.loss_func = loss_func if loss_func else self.default_loss
    def convolve(self, b, t):
        return b * t
    def forward(self, x, u):
        x, u = x.to(self.device), u.to(self.device)
        b = self.branch(u)
        t = self.trunk(x)
        return self.convolve(b, t)
    def predict(self, x, u, printt=False):
        x, u, = torch.tensor(x, dtype=torch.float32).to(self.device), torch.tensor(u, dtype=torch.float32).to(self.device)
        start_time = time.monotonic()
        yp = torch.matmul(self.forward(x, u), self.A.to(self.device)) + self.B.to(self.device)
        elapsed_time = time.monotonic() - start_time
        if printt:
            print(f"Prediction completed in {elapsed_time:.6f} seconds.")
        return yp.detach().cpu().numpy()
    def default_loss(self, y, yp):
        y, yp = y.to(self.device), yp.to(self.device)
        nmse = torch.mean((y - yp) ** 2) / torch.std(y) ** 2
        return nmse
    def loss(self, y, yp):
        return self.loss_func(y, yp)


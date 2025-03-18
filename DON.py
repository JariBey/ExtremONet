import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch

class NeuralNet(nn.Module):
    def __init__(self, indim, layers, outdim, acfuncs, device=None, norm=[0, 1]):
        super(NeuralNet, self).__init__()
        self.device=device
        self.dtype = torch.float32
        self.indim = indim
        self.layers = layers
        self.outdim = outdim
        self.acfuncs = acfuncs
        self.mods = [nn.Linear(indim, layers[0], bias=True)] 
        self.mods += [nn.Linear(layers[i], layers[i+1],bias=True) for i in range(len(layers)-1)] 
        self.mods += [nn.Linear(layers[-1], outdim,bias=True)]
        self.mods = nn.ModuleList(self.mods)
        self.norm = norm
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.mods:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Glorot initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def scale(self, x):
        return (x - self.norm[0]) / self.norm[1]
    def forward(self, x):
        for i in range(len(self.mods)):
            x = self.acfuncs[i](self.mods[i](x))
        return x
    def predict(self, x):
        xt = torch.tensor(x, dtype=self.dtype).to(self.device)
        return self.forward(xt).detach().cpu().numpy()

class DeepONet(nn.Module):
    def __init__(self, branch, trunk, psize, outdim, loss_func=None, device=None):
        super(DeepONet, self).__init__()
        self.device=device
        self.loss_func = loss_func if loss_func else self.default_loss
        self.dtype = torch.float32
        self.branch = branch.to(self.device)
        self.trunk = trunk.to(self.device)
        self.psize = psize
        self.outdim = outdim
        self.bias = nn.Parameter(torch.zeros(1, outdim), requires_grad=True)
        self.paramlist = nn.ParameterList([self.bias])
    def readout(self, b, t):
        b = b.view(b.shape[0], self.psize, self.outdim)
        t = t.view(t.shape[0], self.psize, self.outdim)
        return (b*t).sum(1) + self.bias
    def forward(self, x, u):
        x, u = x.to(self.device), u.to(self.device)
        b = self.branch(u)
        t = self.trunk(x)
        return self.readout(b, t)
    def predict(self, x, u):
        xt = torch.tensor(x, dtype=self.dtype).to(self.device)
        ut = torch.tensor(u, dtype=self.dtype).to(self.device)
        return self.forward(xt, ut).detach().cpu().numpy()
    def default_loss(self, y, yp):
        y, yp = y.to(self.device), yp.to(self.device)
        nmse = torch.mean((y - yp) ** 2) / torch.std(y) ** 2
        return nmse
    def loss(self, y, yp):
        return self.loss_func(y, yp)

class UnconstrainedDeepONet(nn.Module):
    def __init__(self, branch, trunk, latdim, outdim, loss_func=None, device=None):
        super(UnconstrainedDeepONet, self).__init__()
        self.device=device
        self.loss_func = loss_func if loss_func else self.default_loss
        self.dtype = torch.float32
        self.branch = branch.to(self.device)
        self.trunk = trunk.to(self.device)
        self.latdim = latdim
        self.outdim = outdim
        self.bias = nn.Parameter(torch.randn(1, outdim), requires_grad=True)
        self.down = nn.Parameter(torch.rand(latdim, outdim)*2-1, requires_grad=True)
        self.paramlist = nn.ParameterList([self.bias,self.down])
    def readout(self, b, t):
        return (b * t) @ self.down + self.bias
    def forward(self, x, u):
        x, u = x.to(self.device), u.to(self.device)
        b = self.branch(u)
        t = self.trunk(x)
        return self.readout(b, t)
    def predict(self, x, u):
        xt = torch.tensor(x, dtype=self.dtype).to(self.device)
        ut = torch.tensor(u, dtype=self.dtype).to(self.device)
        return self.forward(xt, ut).detach().cpu().numpy()
    def default_loss(self, y, yp):
        y, yp = y.to(self.device), yp.to(self.device)
        nmse = torch.mean((y - yp) ** 2) / torch.std(y) ** 2
        return nmse
    def loss(self, y, yp):
        return self.loss_func(y, yp)

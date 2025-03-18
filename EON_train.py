import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import time
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar,minimize
import matplotlib.pyplot as plt
def train_EON(model, x, u, y, bounds=[-10,10], iters=200, tp=0.2, verbose=True):
    x, u, y = torch.tensor(x, dtype=torch.float32).to(model.device), torch.tensor(u, dtype=torch.float32).to(model.device), torch.tensor(y, dtype=torch.float32).to(model.device)
    start_time = time.time()

    def train_test_split(x, y, perc=0.01):
        inds = np.arange(x.shape[0])
        teinds = np.random.choice(inds, int(perc * x.shape[0]), replace=False)
        trinds = np.array([i for i in inds if i not in teinds])
        xtr, ytr = x[trinds], y[trinds]
        xte, yte = x[teinds], y[teinds]
        return xtr, ytr, xte, yte

    h = model.forward(x, u)
    htr, ytr, hte, yte = train_test_split(h, y, tp)
    htr_aug = torch.cat([htr, torch.ones(htr.shape[0], 1, device=model.device)], dim=1)
    hte_aug = torch.cat([hte, torch.ones(hte.shape[0], 1, device=model.device)], dim=1)
    httr_aug = htr_aug.T

    def ridge(l):
        a = torch.matmul(httr_aug, htr_aug)
        b = 10.0 ** l * torch.eye(a.shape[0], device=a.device)
        Q, R = torch.linalg.qr(a + b)
        c = torch.inverse(R) @ Q.t()
        d = torch.matmul(httr_aug, ytr)
        A_aug = torch.matmul(c, d)
        A = A_aug[:-1, :]
        B = A_aug[-1, :]
        err = model.loss(yte, torch.matmul(hte_aug, A_aug))
        errtr = model.loss(ytr, torch.matmul(htr_aug, A_aug))
        return A, B, err, errtr

    def objective(l):
        A, B, e, etr = ridge(l)
        Ahist.append(A)
        Bhist.append(B)
        ehist.append(e.item())
        etrhist.append(etr.item())
        lhist.append(l)
        return e.item()

    Ahist, Bhist, ehist, etrhist, lhist = [], [], [], [], []
    initial_guess = [bounds[1]]
    result = minimize_scalar(objective,bounds=bounds,method='bounded',options={'maxiter':iters,'xatol':0})#minimize(objective, x0=initial_guess, bounds=[bounds], method='COBYQA', options={ 'f_target': 0, 'feasibility_tol': 0, 'maxfev': iters})
    minind = torch.argmin(torch.tensor(ehist))
    model.A.data = Ahist[minind]
    model.B.data = Bhist[minind]
    model.valloss = torch.min(torch.tensor(ehist))
    end_time = time.time()
    duration = end_time - start_time
    if verbose:
        print('---------------------------------------------------------------------------------------------------')
        print(result)
        print('related train NMSE = ' + str(etrhist[minind]))
        print(f"Training completed in {duration:.2f} seconds.")

    return etrhist,ehist

def basis_declutter(model, x, u, y, perc,iters,tp=0.1):
    def keep(x, y, z,perc=0.1):
        inds = np.arange(x.shape[0])
        teinds = np.random.choice(inds, int(perc * x.shape[0]), replace=False)
        return x[teinds],y[teinds],z[teinds]
    x, u, y = torch.tensor(x, dtype=torch.float32).to(model.device), torch.tensor(u, dtype=torch.float32).to(model.device), torch.tensor(y, dtype=torch.float32).to(model.device)
    x, u, y = keep(x.to(model.device), u.to(model.device), y.to(model.device),tp)
    def score(h, y):
        try:
            print(h.shape,y.shape)
            htr = (h - torch.mean(h, dim=0)) / torch.std(h, dim=0)
            ytr = (y - torch.mean(y, dim=0)) / torch.std(y, dim=0)
            a = torch.matmul(htr.T, htr)
            b = 1e-5 * torch.eye(a.shape[0], device=a.device)
            Q, R = torch.linalg.qr(a + b)
            c = torch.inverse(R) @ Q.t()
            d = torch.matmul(htr.T, ytr)
            A = torch.matmul(c, d)
            Ac = torch.sum(torch.abs(A), dim=1)
            return Ac / torch.sum(Ac)
        except Exception as e:
            print(f"An error occurred: {e}")
            return torch.ones(model.psize)
    for i in range(iters):
        h = model.forward(x, u)
        sco = score(h, y)
        rem_inds = torch.argsort(sco,descending=False)[:int(perc*len(sco))]
        model.branch.R[:,rem_inds] = model.branch.init_R(model.branch.indim,len(rem_inds),model.branch.c).to( model.branch.device)
        model.trunk.R[:,rem_inds] = model.trunk.init_R(model.trunk.indim,len(rem_inds),model.trunk.c).to( model.trunk.device)
        model.branch.b[:,rem_inds] = torch.randn(1, len(rem_inds)).to( model.branch.device)
        model.trunk.b[:,rem_inds] =torch.randn(1, len(rem_inds)).to( model.trunk.device)

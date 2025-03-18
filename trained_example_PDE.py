from EON import *
from DON import *
from PDE import *
from EON_train import *
from DON_train import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from skopt import gp_minimize
from skopt.space import Real, Integer
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb
import matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Set matplotlib style options for LaTeX-like appearance
matplotlib.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

width = 5.5048                   # LaTeX text width (inches)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NMSE(y, yp):
    return np.mean((y - yp)**2) / np.mean(y**2)

def reduce(data, p=1):
    """Reduce data to a given percentage."""
    return data[: int(len(data) * p)]

# Load and reduce datasets
PDE_param_ttr, PDE_param_ytr, PDE_param_utr = [reduce(d,  1) for d in open_data("train_data_KG_PDE.pkl")]
PDE_GRF_ttr, PDE_GRF_ytr, PDE_GRF_utr = [reduce(d,  1) for d in open_data("train_data_diff_PDE.pkl")]

PDE_param_tte, PDE_param_yte, PDE_param_ute = [reduce(d) for d in open_data("test_data_KG_PDE.pkl")]
PDE_GRF_tte, PDE_GRF_yte, PDE_GRF_ute = [reduce(d) for d in open_data("test_data_diff_PDE.pkl")]

PDE_param_ttex, PDE_param_ytex, PDE_param_utex = [reduce(d) for d in open_data("example_KG_PDE.pkl")]
PDE_GRF_ttex, PDE_GRF_ytex, PDE_GRF_utex = [reduce(d) for d in open_data("example_diff_PDE.pkl")]

x_grid_1 = open_data('x_grid_KG_PDE.pkl')
x_grid_2 = open_data('x_grid_PDE.pkl')

# Track best model and best error found during optimization
best_test_error = np.inf
best_model = None

space = [
    Real(1e-5, 10, name='s1'),
    Real(1e-5, 10, name='s2'),
    Integer(1, 300, name='c2')
]
dim = 1000

def objective(params):
    global best_test_error, best_model
    s1, s2, c2 = params
    print('----------------------------------------------------------------------------------------------------------------')
    print(f"Testing parameters: s1={s1}, s2={s2}, c2={c2}")
    # Build ExtremeLearning networks with current hyperparameters
    trunk = ExtremeLearning(
        1, dim, c=1, s=s1, acfunc=nn.Tanh(),
        norm=[np.mean(PDE_param_ttr), np.std(PDE_param_ttr)], device=device
    ).to(device)
    branch = ExtremeLearning(
        PDE_param_utr.shape[1], dim, c=3, s=s2, acfunc=nn.Tanh(),
        norm=[np.mean(PDE_param_utr), np.std(PDE_param_utr)], device=device
    ).to(device)
    model = ExtremONet(PDE_param_ytr.shape[1], dim, trunk, branch, device=device).to(device)
    
    # Train the model; train_EON returns (train_history, test_history)
    train_hist, test_hist = train_EON(model, PDE_param_ttr, PDE_param_utr, PDE_param_ytr, verbose=False)
    test_error = np.min(test_hist)
    print(f"Final test error for current parameters: {test_error}")
    
    # Update the best model if the current one is better
    if test_error < best_test_error:
        best_test_error = test_error
        best_model = model
        print(f"New best model found with error: {best_test_error}")
    return test_error

# Perform Bayesian optimization with gp_minimize
result = gp_minimize(objective, space, n_calls=100)
s1_best, s2_best, c2_best = result.x
print(f"Best parameters: s1={s1_best}, s2={s2_best}, c2={c2_best} with test error {result.fun}")

# Use the best model tracked during optimization for prediction
ypr1 = best_model.predict(PDE_param_ttex, PDE_param_utex)

def objective(params):
    global best_test_error, best_model
    s1, s2, c2 = params
    print('----------------------------------------------------------------------------------------------------------------')
    print(f"Testing parameters: s1={s1}, s2={s2}, c2={c2}")
    # Build ExtremeLearning networks with current hyperparameters
    trunk = ExtremeLearning(
        1, dim, c=1, s=s1, acfunc=nn.Tanh(),
        norm=[np.mean(PDE_GRF_ttr), np.std(PDE_GRF_ttr)], device=device
    ).to(device)
    branch = ExtremeLearning(
        PDE_GRF_utr.shape[1], dim, c=3, s=s2, acfunc=nn.Tanh(),
        norm=[np.mean(PDE_GRF_utr), np.std(PDE_GRF_utr)], device=device
    ).to(device)
    model = ExtremONet(PDE_GRF_ytr.shape[1], dim, trunk, branch, device=device).to(device)
    
    # Train the model; train_EON returns (train_history, test_history)
    train_hist, test_hist = train_EON(model, PDE_GRF_ttr, PDE_GRF_utr, PDE_GRF_ytr, verbose=False)
    test_error = np.min(test_hist)
    print(f"Final test error for current parameters: {test_error}")
    
    # Update the best model if the current one is better
    if test_error < best_test_error:
        best_test_error = test_error
        best_model = model
        print(f"New best model found with error: {best_test_error}")
    return test_error

# Perform Bayesian optimization with gp_minimize
result = gp_minimize(objective, space, n_calls=100)
s1_best, s2_best, c2_best = result.x
print(f"Best parameters: s1={s1_best}, s2={s2_best}, c2={c2_best} with test error {result.fun}")

# Use the best model tracked during optimization for prediction
ypr2 = best_model.predict(PDE_GRF_ttex, PDE_GRF_utex)

import matplotlib.gridspec as gridspec

# Create a figure with a GridSpec that reserves a narrow column for colorbars
fig = plt.figure(figsize=(width * 0.8, width * 0.5))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.1, hspace=0.5)

# First row: two plots and one colorbar axis
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
cbax0 = fig.add_subplot(gs[0, 2])

# Second row: two plots and one colorbar axis
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
cbax1 = fig.add_subplot(gs[1, 2])

# First row using x_grid_1 and PDE_param_ttex
X, T = np.meshgrid(x_grid_1.reshape(-1, 1), PDE_param_ttex)
pc1 = ax00.contourf(T, X, PDE_param_ytex, cmap='ocean', levels=50)
ax00.set_title('a)')
ax00.set_yticks([])
mid_value = np.median(x_grid_1)
ax00.axhline(mid_value, color='black', linewidth=1)

pc2 = ax01.contourf(T, X, ypr1, cmap='ocean', levels=pc1.levels)
ax01.set_title('b)')
ax01.set_yticks([])
ax01.axhline(mid_value, color='black', linewidth=1)

# Add a shared colorbar for the first row to the right
fig.colorbar(pc1, cax=cbax0)

# Second row using x_grid_2 and PDE_GRF_ttex
X2, T2 = np.meshgrid(x_grid_2.reshape(-1, 1), PDE_GRF_ttex)
pc3 = ax10.contourf(T2, X2, PDE_GRF_ytex, cmap='hot', levels=50)
ax10.set_title('c)')
ax10.set_yticks([])

pc4 = ax11.contourf(T2, X2, ypr2, cmap='hot', levels=pc3.levels)
ax11.set_title('d)')
ax11.set_yticks([])

# Add a shared colorbar for the second row to the right
fig.colorbar(pc3, cax=cbax1)
fig.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1)
plt.savefig("example_PDE.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()
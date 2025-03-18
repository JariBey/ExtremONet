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
ODE_param_ttr, ODE_param_ytr, ODE_param_utr = [reduce(d, 1) for d in open_data("train_data_ODE_L63.pkl")]
ODE_GRF_ttr, ODE_GRF_ytr, ODE_GRF_utr = [reduce(d,  1) for d in open_data("train_data_ODE_ut.pkl")]

ODE_param_tte, ODE_param_yte, ODE_param_ute = [reduce(d) for d in open_data("test_data_ODE_L63.pkl")]
ODE_GRF_tte, ODE_GRF_yte, ODE_GRF_ute = [reduce(d) for d in open_data("test_data_ODE_ut.pkl")]

ODE_param_ttex, ODE_param_ytex, ODE_param_utex = [reduce(d) for d in open_data("example_ODE_L63.pkl")]
ODE_GRF_ttex, ODE_GRF_ytex, ODE_GRF_utex = [reduce(d) for d in open_data("example_ODE_ut.pkl")]


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
        norm=[np.mean(ODE_param_ttr), np.std(ODE_param_ttr)], device=device
    ).to(device)
    branch = ExtremeLearning(
        ODE_param_utr.shape[1], dim, c=3, s=s2, acfunc=nn.Tanh(),
        norm=[np.mean(ODE_param_utr), np.std(ODE_param_utr)], device=device
    ).to(device)
    model = ExtremONet(ODE_param_ytr.shape[1], dim, trunk, branch, device=device).to(device)
    
    # Train the model; train_EON returns (train_history, test_history)
    basis_declutter(model, ODE_param_ttr, ODE_param_utr, ODE_param_ytr, perc=0.1, iters=100, tp=0.1)
    train_hist, test_hist = train_EON(model, ODE_param_ttr, ODE_param_utr, ODE_param_ytr, verbose=False)
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
ypr1 = best_model.predict(ODE_param_ttex, ODE_param_utex)

def objective(params):
    global best_test_error, best_model
    s1, s2, c2 = params
    print('----------------------------------------------------------------------------------------------------------------')
    print(f"Testing parameters: s1={s1}, s2={s2}, c2={c2}")
    # Build ExtremeLearning networks with current hyperparameters
    trunk = ExtremeLearning(
        1, dim, c=1, s=s1, acfunc=nn.Tanh(),
        norm=[np.mean(ODE_GRF_ttr), np.std(ODE_GRF_ttr)], device=device
    ).to(device)
    branch = ExtremeLearning(
        ODE_GRF_utr.shape[1], dim, c=3, s=s2, acfunc=nn.Tanh(),
        norm=[np.mean(ODE_GRF_utr), np.std(ODE_GRF_utr)], device=device
    ).to(device)
    model = ExtremONet(ODE_GRF_ytr.shape[1], dim, trunk, branch, device=device).to(device)
    
    # Train the model; train_EON returns (train_history, test_history)
    train_hist, test_hist = train_EON(model, ODE_GRF_ttr, ODE_GRF_utr, ODE_GRF_ytr, verbose=False)
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
ypr2 = best_model.predict(ODE_GRF_ttex, ODE_GRF_utex)

fig,ax = plt.subplots(2,2,figsize=(width*.8,width*.5))

colors1 = plt.cm.Dark2(np.linspace(0, 1, ODE_param_ytex.shape[1]))
colors2 = plt.cm.Paired(np.linspace(0, 1, ODE_GRF_ytex.shape[1]))

for i, col in enumerate(colors1):
    ax[0,0].plot(ODE_param_ttex, ODE_param_ytex[:, i], label=f'True dim {i+1}', color=col, linestyle='-.')
    ax[0,1].plot(ODE_param_ttex, ypr1[:, i], label=f'Predicted dim {i+1}', color=col)
for i, col in enumerate(colors2):
    ax[1,0].plot(ODE_GRF_ttex, ODE_GRF_ytex[:, i], label=f'True dim {i+1}', color=col, linestyle='-.')
    ax[1,1].plot(ODE_GRF_ttex, ypr2[:, i], label=f'Predicted dim {i+1}', color=col)
ax[0,0].set_title("a)")
ax[0,1].set_title("b)")
ax[1,0].set_title("c)")
ax[1,1].set_title("d)")
plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
plt.savefig("example_ODE.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()
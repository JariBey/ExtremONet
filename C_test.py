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
    "font.size": 170,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

width = 5.5048                   # LaTeX text width (inches)
height = width / 1.618           # Corresponding height (inches)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NMSE(y, yp):
    return np.mean((y - yp)**2) / np.mean(y**2)

def reduce(data, p=1):
    """Reduce data to a given percentage."""
    return data[: int(len(data) * p)]

# Load and reduce datasets
ODE_param_ttr, ODE_param_ytr, ODE_param_utr = [reduce(d, 0.1) for d in open_data("train_data_ODE_L63.pkl")]
ODE_GRF_ttr, ODE_GRF_ytr, ODE_GRF_utr = [reduce(d,  0.1) for d in open_data("train_data_ODE_ut.pkl")]
PDE_param_ttr, PDE_param_ytr, PDE_param_utr = [reduce(d,  0.1) for d in open_data("train_data_KG_PDE.pkl")]
PDE_GRF_ttr, PDE_GRF_ytr, PDE_GRF_utr = [reduce(d,  0.1) for d in open_data("train_data_diff_PDE.pkl")]

ODE_param_tte, ODE_param_yte, ODE_param_ute = [reduce(d) for d in open_data("test_data_ODE_L63.pkl")]
ODE_GRF_tte, ODE_GRF_yte, ODE_GRF_ute = [reduce(d) for d in open_data("test_data_ODE_ut.pkl")]
PDE_param_tte, PDE_param_yte, PDE_param_ute = [reduce(d) for d in open_data("test_data_KG_PDE.pkl")]
PDE_GRF_tte, PDE_GRF_yte, PDE_GRF_ute = [reduce(d) for d in open_data("test_data_diff_PDE.pkl")]

ODE_param_ttex, ODE_param_ytex, ODE_param_utex = [reduce(d) for d in open_data("example_ODE_L63.pkl")]
ODE_GRF_ttex, ODE_GRF_ytex, ODE_GRF_utex = [reduce(d) for d in open_data("example_ODE_ut.pkl")]
PDE_param_ttex, PDE_param_ytex, PDE_param_utex = [reduce(d) for d in open_data("example_KG_PDE.pkl")]
PDE_GRF_ttex, PDE_GRF_ytex, PDE_GRF_utex = [reduce(d) for d in open_data("example_diff_PDE.pkl")]

def test_EON(ttr, utr, ytr, tte, ute, yte, c):
    s1_best, s2_best = 5, 0.1

    trunk_norm = [np.mean(ttr), np.std(ttr)]
    branch_norm = [np.mean(utr), np.std(utr)]
    trunk = ExtremeLearning(1, 1000, c=1, s=s1_best, acfunc=nn.Tanh(), norm=trunk_norm, device=device).to(device)
    branch = ExtremeLearning(utr.shape[1], 1000, c=c, s=s2_best, acfunc=nn.Tanh(), norm=branch_norm, device=device).to(device)
    EON = ExtremONet(ytr.shape[1], 1000, trunk, branch, device=device).to(device)
    start_time = time.time()
    trhist, valhist = train_EON(EON, ttr, utr, ytr, iters=100)
    trtime = time.time() - start_time
    start_time = time.time()
    ytep = EON.predict(tte, ute)
    predtime = time.time() - start_time
    ytrp = EON.predict(ttr, utr)
    del trunk, branch, EON
    torch.cuda.empty_cache()
    return NMSE(ytr, ytrp), NMSE(yte, ytep), trtime, predtime/tte.shape[0]

def testloop(ttr, utr, ytr, tte, ute, yte, repeats, sensors):
    Etrhist_mean, Evalhist_mean = [], []
    Etrtime_mean, Epredtime_mean = [], []
    Etrhist_std, Evalhist_std = [], []
    Etrtime_std, Epredtime_std = [], []

    for sensor in sensors:
        sEtrhist, sEvalhist, sEtrtime, sEpredtime = [], [], [], []
        for _ in range(repeats):
            res = test_EON(ttr, utr, ytr, tte, ute, yte, sensor)
            sEtrhist.append(res[0])
            sEvalhist.append(res[1])
            sEtrtime.append(res[2])
            sEpredtime.append(res[3])
        Etrhist_mean.append(np.mean(sEtrhist))
        Evalhist_mean.append(np.mean(sEvalhist))
        Etrtime_mean.append(np.mean(sEtrtime))
        Epredtime_mean.append(np.mean(sEpredtime))
        Etrhist_std.append(np.std(sEtrhist))
        Evalhist_std.append(np.std(sEvalhist))
        Etrtime_std.append(np.std(sEtrtime))
        Epredtime_std.append(np.std(sEpredtime))
    return (Etrhist_mean, Evalhist_mean, Etrtime_mean, Epredtime_mean,
            Etrhist_std, Evalhist_std, Etrtime_std, Epredtime_std)

nrepeat = 30
sensors = np.linspace(1, 300, 20, dtype=int)

eth_ODE_param_mean, evh_ODE_param_mean, ett_ODE_param_mean, ept_ODE_param_mean, \
eth_ODE_param_std, evh_ODE_param_std, ett_ODE_param_std, ept_ODE_param_std = testloop(
    ODE_param_ttr, ODE_param_utr, ODE_param_ytr,
    ODE_param_tte, ODE_param_ute, ODE_param_yte, nrepeat, sensors)

eth_ODE_GRF_mean, evh_ODE_GRF_mean, ett_ODE_GRF_mean, ept_ODE_GRF_mean, \
eth_ODE_GRF_std, evh_ODE_GRF_std, ett_ODE_GRF_std, ept_ODE_GRF_std = testloop(
    ODE_GRF_ttr, ODE_GRF_utr, ODE_GRF_ytr,
    ODE_GRF_tte, ODE_GRF_ute, ODE_GRF_yte, nrepeat, sensors)

eth_PDE_param_mean, evh_PDE_param_mean, ett_PDE_param_mean, ept_PDE_param_mean, \
eth_PDE_param_std, evh_PDE_param_std, ett_PDE_param_std, ept_PDE_param_std = testloop(
    PDE_param_ttr, PDE_param_utr, PDE_param_ytr,
    PDE_param_tte, PDE_param_ute, PDE_param_yte, nrepeat, sensors)

eth_PDE_GRF_mean, evh_PDE_GRF_mean, ett_PDE_GRF_mean, ept_PDE_GRF_mean, \
eth_PDE_GRF_std, evh_PDE_GRF_std, ett_PDE_GRF_std, ept_PDE_GRF_std = testloop(
    PDE_GRF_ttr, PDE_GRF_utr, PDE_GRF_ytr,
    PDE_GRF_tte, PDE_GRF_ute, PDE_GRF_yte, nrepeat, sensors)

# Define mosaic layout for subplots
mosaic = "ab\ncd"
fig, axs = plt.subplot_mosaic(mosaic, figsize=(width * 0.8, width * 0.5))

# Prepare data to plot
data = [((eth_ODE_param_mean, eth_ODE_param_std), (evh_ODE_param_mean, evh_ODE_param_std)),
        ((eth_ODE_GRF_mean, eth_ODE_GRF_std), (evh_ODE_GRF_mean, evh_ODE_GRF_std)),
        ((eth_PDE_param_mean, eth_PDE_param_std), (evh_PDE_param_mean, evh_PDE_param_std)),
        ((eth_PDE_GRF_mean, eth_PDE_GRF_std), (evh_PDE_GRF_mean, evh_PDE_GRF_std))]

base_colors = {"ab": "blue"}
def generate_shades(base_color, num_shades=2):
    rgb = to_rgb(base_color)
    return [tuple(x * factor for x in rgb) for factor in np.linspace(0.3, 1, num_shades)]

ab_shades = generate_shades(base_colors["ab"], num_shades=len(sensors))
linestyles = {"a": "-", "b": "-."}

def calculate_percentile(*data):
    all_data = np.concatenate([np.concatenate([np.array(di).flatten() for di in d]) for d in data])
    return np.percentile(all_data, 99)

def calculate_min(*data):
    all_data = np.concatenate([np.concatenate([np.array(di).flatten() for di in d]) for d in data])
    return np.min(all_data)

def set_log_ticks(ax, y_min, y_max):
    exp_min = int(np.floor(np.log10(y_min))) if y_min > 0 else 0
    exp_max = int(np.ceil(np.log10(y_max)))
    ticks = [10**i for i in range(exp_min, exp_max + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{i}}}$" for i in range(exp_min, exp_max + 1)])

# Plot subplots
for (key, ax), (eth, evh) in zip(axs.items(), data):
    y_max = calculate_percentile(np.array(eth[0]) + np.array(eth[1]), 
                                 np.array(evh[0]) + np.array(evh[1]))
    y_min = calculate_min(np.array(eth[0]) - np.array(eth[1]), 
                          np.array(evh[0]) - np.array(evh[1]))
    ax.set_yscale("log")
    ax.set_ylim((y_min, y_max))
    set_log_ticks(ax, y_min, y_max)

    ax.plot(sensors, eth[0], color="blue", linestyle=linestyles["a"], label="a")
    ax.fill_between(sensors, np.array(eth[0]) - np.array(eth[1]), 
                    np.array(eth[0]) + np.array(eth[1]), color="blue", alpha=0.2)
    ax.plot(sensors, evh[0], color="black", linestyle=linestyles["b"], label="b")
    ax.fill_between(sensors, np.array(evh[0]) - np.array(evh[1]), 
                    np.array(evh[0]) + np.array(evh[1]), color="black", alpha=0.2)
    ax.set_title( f"{key})")
    ax.grid()

plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
plt.savefig("Cb_comp.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)

mean_ptE = np.mean((ept_ODE_GRF_mean, ept_ODE_param_mean, ept_PDE_GRF_mean, ept_PDE_param_mean), axis=0)
mean_ttE = np.mean((ett_ODE_GRF_mean, ett_ODE_param_mean, ett_PDE_GRF_mean, ett_PDE_param_mean), axis=0)

bar_width = (sensors[1] - sensors[0]) * 0.8
fig, axs = plt.subplots(1, 2, figsize=(width * 0.8, width * 0.3))

# Train Time Plot
axs[0].bar(sensors, mean_ttE, yerr=np.std(mean_ttE, axis=0), width=bar_width, capsize=2, 
           label="EON", alpha=0.7, color="blue")
axs[0].set_ylabel(r"T.T.(log10 s)")
exp_min = int(np.floor(np.log10(np.min(mean_ttE))))
exp_max = int(np.ceil(np.log10(np.max(mean_ttE))))
ticks = [10**i for i in range(exp_min, exp_max + 1)]
axs[0].set_yscale("log")
axs[0].set_yticks(ticks)
axs[0].set_yticklabels([f"$10^{{{i}}}$" for i in range(exp_min, exp_max + 1)])
axs[0].set_title( f"a)")

# Prediction Time Plot
axs[1].bar(sensors, mean_ptE, yerr=np.std(mean_ptE, axis=0), width=bar_width, capsize=2, 
           label="EON", alpha=0.7, color="blue")
axs[1].set_ylabel(r'P.T./$n$(log10 s)')
exp_min = int(np.floor(np.log10(np.min(mean_ptE))))
exp_max = int(np.ceil(np.log10(np.max(mean_ptE))))
ticks = [10**i for i in range(exp_min, exp_max + 1)]
axs[1].set_yscale("log")
axs[1].set_yticks(ticks)
axs[1].set_yticklabels([f"$10^{{{i}}}$" for i in range(exp_min, exp_max + 1)])
axs[1].set_title( f"b)")

plt.tight_layout()
fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, wspace=0.6)
plt.savefig("Cb_time_comp.svg", dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()

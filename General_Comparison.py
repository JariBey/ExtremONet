import os, time, numpy as np, matplotlib.pyplot as plt, matplotlib
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from skopt import gp_minimize
from skopt.space import Real, Integer
import torch, torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from matplotlib.colors import to_rgb
from EON import *
from DON import *
from PDE import *
from EON_train import *
from DON_train import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

width = 5.5048                # LaTeX text width in inches
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reduce(data, p=1):
    return data[: int(len(data) * p)]

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

def test_DON(ttr, utr, ytr, tte, ute, yte, layers):
    iters = 100000
    afs = [nn.Tanh() if i != len(layers) else lambda x: x for i in range(len(layers) + 1)]
    Trunk = NeuralNet(ttr.shape[1], layers, ytr.shape[1] * 10, afs, device=device, norm=[np.mean(ttr), np.std(ttr)]).to(device)
    Branch = NeuralNet(utr.shape[1], layers, ytr.shape[1] * 10, afs, device=device, norm=[np.mean(utr), np.std(utr)]).to(device)
    DON_model = DeepONet(Branch, Trunk, 10, ytr.shape[1], device=device).to(device)
    print(f"Total parameters in DON: {sum(p.numel() for p in DON_model.parameters() if p.requires_grad)}")
    optim = AdamW(DON_model.parameters(), weight_decay=1e-3)
    scheduler = lr_scheduler.LinearLR(optim, 1e-3, 1e-3, iters)
    start_time = time.time()
    trhist, valhist = train_DON(DON_model, optim, scheduler, [ttr, utr, ytr], iters)
    trtime = time.time() - start_time
    start_time = time.time()
    _ = DON_model.predict(tte, ute)
    predtime = time.time() - start_time
    del Trunk, Branch, DON_model
    torch.cuda.empty_cache()
    return trhist, valhist, trtime, predtime

def test_EON(ttr, utr, ytr, tte, ute, yte, width):
    trunk = ExtremeLearning(1, width, c=1, s=5, acfunc=nn.Tanh(), norm=[np.mean(ttr), np.std(ttr)], device=device).to(device)
    branch = ExtremeLearning(utr.shape[1], width, c=3, s=0.1, acfunc=nn.Tanh(), norm=[np.mean(utr), np.std(utr)], device=device).to(device)
    EON_model = ExtremONet(ytr.shape[1], width, trunk, branch, device=device).to(device)
    start_time = time.time()
    trhist, valhist = train_EON(EON_model, ttr, utr, ytr, iters=100)
    trtime = time.time() - start_time
    start_time = time.time()
    _ = EON_model.predict(tte, ute)
    predtime = time.time() - start_time
    del trunk, branch, EON_model
    torch.cuda.empty_cache()
    return trhist, valhist, trtime, predtime

def test_EON_HPS(ttr, utr, ytr, tte, ute, yte, width):
    best_test_error = np.inf
    best_model = None
    trhist = []; valhist = []
    space = [
        Real(1e-5, 10, name='s1'),
        Real(1e-5, 10, name='s2'),
        Integer(1, 100, name='c2')
    ]
    dim = 1000
    def objective(params):
        global best_test_error, best_model
        s1, s2, c2 = params
        print('----------------------------------------------------------------------------------------------------------------')
        print(f"Testing parameters: s1={s1}, s2={s2}, c2={c2}")
        trunk = ExtremeLearning(
            1, width, c=1, s=s1, acfunc=nn.Tanh(),
            norm=[np.mean(PDE_param_ttr), np.std(PDE_param_ttr)], device=device
        ).to(device)
        branch = ExtremeLearning(
            PDE_param_utr.shape[1], width, c=3, s=s2, acfunc=nn.Tanh(),
            norm=[np.mean(PDE_param_utr), np.std(PDE_param_utr)], device=device
        ).to(device)
        model = ExtremONet(PDE_param_ytr.shape[1], width, trunk, branch, device=device).to(device)
        
        train_hist, test_hist = train_EON(model, PDE_param_ttr, PDE_param_utr, PDE_param_ytr, verbose=False)
        test_error = np.min(test_hist)
        print(f"Final test error for current parameters: {test_error}")
        trhist+=train_hist
        valhist+=test_hist
        if test_error < best_test_error:
            best_test_error = test_error
            best_model = model
            print(f"New best model found with error: {best_test_error}")
        return test_error
    start_time = time.time()
    result = gp_minimize(objective, space, n_calls=10)
    s1_best, s2_best, c2_best = result.x
    print(f"Best parameters: s1={s1_best}, s2={s2_best}, c2={c2_best} with test error {result.fun}")
    trtime = time.time() - start_time
    start_time = time.time()
    _ = best_model.predict(tte, ute)
    predtime = time.time() - start_time
    del trunk, branch, model
    torch.cuda.empty_cache()
    return trhist, valhist, trtime, predtime

def testloop(ttr, utr, ytr, tte, ute, yte, repeats, layers, widths):
    Dtrhist, Dvalhist, Dtrtime, Dpredtime = [], [], [], []
    Etrhist, Evalhist, Etrtime, Epredtime = [], [], [], []
    for layer, width in zip(layers, widths):
        sDtrhist = []; sDvalhist = []; sDtrtime = []; sDpredtime = []
        sEtrhist = []; sEvalhist = []; sEtrtime = []; sEpredtime = []
        for _ in range(repeats):
            d_res = test_DON(ttr, utr, ytr, tte, ute, yte, layer)
            e_res = test_EON(ttr, utr, ytr, tte, ute, yte, width)
            sDtrhist.append(d_res[0]); sDvalhist.append(d_res[1])
            sDtrtime.append(d_res[2]); sDpredtime.append(d_res[3])
            sEtrhist.append(e_res[0]); sEvalhist.append(e_res[1])
            sEtrtime.append(e_res[2]); sEpredtime.append(e_res[3])
        # Pad and average histories
        max_len_eon_train = max(len(hist) for hist in sEtrhist)
        max_len_eon_test = max(len(hist) for hist in sEvalhist)
        max_len_don_train = max(len(hist) for hist in sDtrhist)
        max_len_don_test = max(len(hist) for hist in sDvalhist)
        sEtrhist = [np.pad(hist, (0, max_len_eon_train - len(hist)), 'edge') for hist in sEtrhist]
        sEvalhist = [np.pad(hist, (0, max_len_eon_test - len(hist)), 'edge') for hist in sEvalhist]
        sDtrhist = [np.pad(hist, (0, max_len_don_train - len(hist)), 'edge') for hist in sDtrhist]
        sDvalhist = [np.pad(hist, (0, max_len_don_test - len(hist)), 'edge') for hist in sDvalhist]
        Dtrhist.append(np.mean(sDtrhist, axis=0))
        Dvalhist.append(np.mean(sDvalhist, axis=0))
        Dtrtime.append(np.mean(sDtrtime))
        Dpredtime.append(np.mean(sDpredtime))
        Etrhist.append(np.mean(sEtrhist, axis=0))
        Evalhist.append(np.mean(sEvalhist, axis=0))
        Etrtime.append(np.mean(sEtrtime))
        Epredtime.append(np.mean(sEpredtime))
    return Dtrhist, Dvalhist, Dtrtime, Dpredtime, Etrhist, Evalhist, Etrtime, Epredtime

nrepeat = 1
layers = [[100], [200, 200]]
widths = [100, 1000]

(dth_ODE_param, dvh_ODE_param, dtt_ODE_param, dpt_ODE_param,
 eth_ODE_param, evh_ODE_param, ett_ODE_param, ept_ODE_param) = testloop(
    ODE_param_ttr, ODE_param_utr, ODE_param_ytr, ODE_param_tte, ODE_param_ute, ODE_param_yte,
    nrepeat, layers, widths
)
(dth_ODE_GRF, dvh_ODE_GRF, dtt_ODE_GRF, dpt_ODE_GRF,
 eth_ODE_GRF, evh_ODE_GRF, ett_ODE_GRF, ept_ODE_GRF) = testloop(
    ODE_GRF_ttr, ODE_GRF_utr, ODE_GRF_ytr, ODE_GRF_tte, ODE_GRF_ute, ODE_GRF_yte,
    nrepeat, layers, widths
)
(dth_PDE_param, dvh_PDE_param, dtt_PDE_param, dpt_PDE_param,
 eth_PDE_param, evh_PDE_param, ett_PDE_param, ept_PDE_param) = testloop(
    PDE_param_ttr, PDE_param_utr, PDE_param_ytr, PDE_param_tte, PDE_param_ute, PDE_param_yte,
    nrepeat, layers, widths
)
(dth_PDE_GRF, dvh_PDE_GRF, dtt_PDE_GRF, dpt_PDE_GRF,
 eth_PDE_GRF, evh_PDE_GRF, ett_PDE_GRF, ept_PDE_GRF) = testloop(
    PDE_GRF_ttr, PDE_GRF_utr, PDE_GRF_ytr, PDE_GRF_tte, PDE_GRF_ute, PDE_GRF_yte,
    nrepeat, layers, widths
)

# Plotting: mosaic layout for performance histories
mosaic = """
    ab
    cd
"""
fig, axs = plt.subplot_mosaic(mosaic, figsize=(width * 0.8, width * 0.5))
data = [
    (eth_ODE_param, evh_ODE_param, dth_ODE_param, dvh_ODE_param),
    (eth_ODE_GRF,   evh_ODE_GRF,   dth_ODE_GRF,   dvh_ODE_GRF),
    (eth_PDE_param, evh_PDE_param, dth_PDE_param, dvh_PDE_param),
    (eth_PDE_GRF,   evh_PDE_GRF,   dth_PDE_GRF,   dvh_PDE_GRF),
]
base_colors = {'ab': 'blue', 'cd': 'red'}
def generate_shades(base_color, num_shades=2):
    rgb = to_rgb(base_color)
    return [tuple(x * f for x in rgb) for f in np.linspace(0.4, 1, num_shades)]
ab_shades, cd_shades = generate_shades(base_colors['ab']), generate_shades(base_colors['cd'])
linestyles = {'a': '-', 'b': '-.', 'c': '-', 'd': '-.'}

def calculate_percentile(*data):
    d = np.concatenate([np.concatenate([np.array(di).flatten() for di in dset]) for dset in data])
    return np.percentile(d, 99)

def calculate_min(*data):
    d = np.concatenate([np.concatenate([np.array(di).flatten() for di in dset]) for dset in data])
    return np.min(d)

def set_log_ticks(ax, y_min, y_max):
    exp_min = int(np.floor(np.log10(y_min))) if y_min > 0 else 0
    exp_max = int(np.ceil(np.log10(y_max)))
    ticks = [10**i for i in range(exp_min, exp_max + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{i}}}$" for i in range(exp_min, exp_max + 1)])

for (key, ax), (eth, evh, dth, dvh) in zip(axs.items(), data):
    y_max = calculate_percentile(eth, evh, dth, dvh)
    y_min = calculate_min(eth, evh, dth, dvh)
    ax.set_yscale('log'); ax.set_ylim((y_min, y_max)); set_log_ticks(ax, y_min, y_max)
    ax_top = ax.twiny(); ax_top.set_yscale('log'); ax_top.set_ylim((y_min, y_max)); set_log_ticks(ax_top, y_min, y_max)
    for (a, b, c, d, shade1, shade2) in zip(eth, evh, dth, dvh, ab_shades, cd_shades):
        ax.plot(range(len(a)), a, color=shade1, linestyle=linestyles['a'],linewidth=1)
        ax.plot(range(len(b)), b, color=shade1, linestyle=linestyles['b'],linewidth=1)
        ax_top.plot(range(len(c)), c, color=shade2, linestyle=linestyles['c'],linewidth=1)
        ax_top.plot(range(len(d)), d, color=shade2, linestyle=linestyles['d'],linewidth=1)
    ax_top.text(-0.07, 1.10, f'{key})', transform=ax_top.transAxes, ha='right', va='bottom')
    ax.grid()
plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
plt.savefig('general_comp.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)
data = [
(dtt_ODE_param, ett_ODE_param, dpt_ODE_param, ept_ODE_param),
(dtt_ODE_GRF, ett_ODE_GRF, dpt_ODE_GRF, ept_ODE_GRF),
(dtt_PDE_param, ett_PDE_param, dpt_PDE_param, ept_PDE_param),
(dtt_PDE_GRF, ett_PDE_GRF, dpt_PDE_GRF, ept_PDE_GRF),
]

fig, axs = plt.subplot_mosaic(mosaic, figsize=(width * 0.8, width * 0.5))
exp_min = int(np.floor(np.log10(np.min(data))))
exp_max = int(np.ceil(np.log10(np.max(data))))
for (key, ax), (ttd, tte, ptd, pte) in zip(axs.items(), data):
    ca,cb = ab_shades
    cc,cd = cd_shades
    x_labels = [f'({i+1})' for i in range(4)]
    pos = np.arange(4)
    width = 0.4
    ax.bar(pos - width/2, np.concatenate((ttd, ptd)),
           width=width, alpha=0.7, color=[cc, cd, cc, cd], linewidth=1)
    ax.bar(pos + width/2, np.concatenate((tte, pte)),
           width=width, alpha=0.7, color=[ca, cb, ca, cb], linewidth=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(x_labels)
    
    ticks = [10**i for i in range(exp_min, exp_max + 1)]
    ax.set_yscale('log'); ax.set_yticks(ticks)
    ax.set_yticklabels([f"$10^{{{i}}}$" for i in range(exp_min, exp_max + 1)])
    ax.set_title( f'{key})')
plt.tight_layout()
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
plt.savefig('time_comp.svg', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()

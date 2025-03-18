import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp  # changed from odeint to solve_ivp
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import pickle

'-----------------------------------------------------------------------------------------------------------'

def rbf_kernel(x1, x2, l):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * l**2))

def generate_grf(domain, l, num_samples=1):
    n = len(domain)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = rbf_kernel(domain[i], domain[j], l)
    mean = np.zeros(n)
    samples = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    return samples

def interpolate_grf(domain, grf_sample):
    return interp1d(domain, grf_sample, kind='quadratic', fill_value="extrapolate")

def ut_ode(y, t, u):
    theta, omega = y
    u_value = u(t)
    dtheta_dt = omega
    domega_dt = - np.sin(theta) + u_value
    return np.array([dtheta_dt, domega_dt])

'-----------------------------------------------------------------------------------------------------------'

def Lorenz63(y, t, params):
    sigma, rho, beta = params
    dydt = [sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2]]
    return np.array(dydt)

'-----------------------------------------------------------------------------------------------------------'

def GetSensors(u, sensor_locs):
    sensor_readings = u(sensor_locs).flatten()
    return sensor_readings

def isnotnan(x):
    return [not (np.any(np.isnan(xi)) or np.any(np.isneginf(xi)) or np.any(np.isposinf(xi))) for xi in x]

def GenerateTimeseries_ODE_GRF(diffeq, y0, Trange, l, sensor_locs, num_samples, grid, save_path=None):
    def generate_sample():
        # Create t_grid with first entry 0, then random times
        t_grid = np.append(np.zeros(1), np.random.uniform(Trange[0], Trange[1], Trange[2]))
        t_grid.sort()
        # Generate GRF and interpolate
        grf_sample = generate_grf(grid, l, num_samples=1)[0]
        u_interp = interpolate_grf(grid, grf_sample)
        # Define a wrapper with correct argument order for solve_ivp
        f = lambda t, y: diffeq(y, t, u_interp)
        sol = solve_ivp(f, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid, method="RK45")
        # Exclude the initial point for consistency with previous code
        t_eval = sol.t[1:]
        y_eval = sol.y.T[1:]
        sensor_vals = GetSensors(u_interp, sensor_locs)
        sensor_vals_rep = np.repeat(sensor_vals.reshape(1, -1), len(t_eval), axis=0)
        return t_eval.reshape(-1, 1), y_eval, sensor_vals_rep

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((np.vstack(ts), np.vstack(ys), np.vstack(us)), f)
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    tnan = isnotnan(ts_array)
    ynan = isnotnan(ys_array)
    unan = isnotnan(us_array)
    nans = tnan and ynan and unan
    ts_array = ts_array[nans]
    ys_array = ys_array[nans]
    us_array = us_array[nans]
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((ts_array, ys_array, us_array), f)
    return ts_array, ys_array, us_array

def GenerateTimeseries_ODE_param(diffeq, y0, Trange, param_range, sensor_locs, num_samples, save_path=None):
    def generate_sample():
        t_grid = np.append(np.zeros(1), np.random.uniform(Trange[0], Trange[1], Trange[2]))
        t_grid.sort()
        param = [np.random.uniform(r[0], r[1]) for r in param_range]
        # Define wrapper to swap arguments for solve_ivp
        f = lambda t, y: diffeq(y, t, param)
        sol = solve_ivp(f, (t_grid[0], t_grid[-1]), y0, t_eval=t_grid, method="RK45")
        t_eval = sol.t[1:]
        y_eval = sol.y.T[1:]
        # For sensor readings, create a lambda that calls diffeq with fixed param
        sensor_vals = GetSensors(lambda x: diffeq(x, 0, param), sensor_locs.T)
        sensor_vals_rep = np.repeat(sensor_vals.reshape(1, -1), len(t_eval), axis=0)
        return t_eval.reshape(-1, 1), y_eval, sensor_vals_rep

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    tnan = isnotnan(ts_array)
    ynan = isnotnan(ys_array)
    unan = isnotnan(us_array)
    nans = tnan and ynan and unan
    ts_array = ts_array[nans]
    ys_array = ys_array[nans]
    us_array = us_array[nans]
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((ts_array, ys_array, us_array), f)
    return ts_array, ys_array, us_array

def y0_func(scale=1,dim=1):
    return np.random.normal(0, scale, dim)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)   

def open_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    
    Trange = [0, 1, 10]
    TrangeL63 = [0, 1, 10]
 
    sensor_locs_ut = np.random.uniform(0, 1, (300, 1))
    sensor_locs_L63 = np.random.uniform(-30, 30, (100, 3))
    Lparamr = [[0, 10.0], [0, 28.0], [0, 8.0/3.0]]

    num_train_samples = 10000
    num_test_samples = 10000
    y0 = y0_func(.1,2)  # Update to match the 2D state vector
    y0L63 = y0_func(1,3)
    gridt = np.linspace(Trange[0], Trange[1], 100)
    lt = 0.1

    GenerateTimeseries_ODE_GRF(ut_ode, y0, Trange, lt, sensor_locs_ut, num_train_samples, gridt, save_path='train_data_ODE_ut.pkl')
    GenerateTimeseries_ODE_GRF(ut_ode, y0, Trange, lt, sensor_locs_ut, num_test_samples, gridt, save_path='test_data_ODE_ut.pkl')
    GenerateTimeseries_ODE_GRF(ut_ode, y0, [Trange[0],Trange[1],1000], lt, sensor_locs_ut, 1, gridt, save_path='example_ODE_ut.pkl')

    GenerateTimeseries_ODE_param(Lorenz63, y0L63, TrangeL63, Lparamr,sensor_locs_L63, num_train_samples, save_path='train_data_ODE_L63.pkl')
    GenerateTimeseries_ODE_param(Lorenz63, y0L63, TrangeL63, Lparamr,sensor_locs_L63, num_test_samples, save_path='test_data_ODE_L63.pkl')
    GenerateTimeseries_ODE_param(Lorenz63, y0L63, [TrangeL63[0],TrangeL63[1],1000], Lparamr,sensor_locs_L63, 1, save_path='example_ODE_L63.pkl')

    save_data(Trange, 'Trange_ODE_ut.pkl')
    save_data(TrangeL63, 'Trange_ODE_L63.pkl')
    save_data(sensor_locs_ut, 'sensor_locs_ODE_ut.pkl')
    save_data(sensor_locs_L63, 'sensor_locs_ODE_L63.pkl')
    save_data(y0, 'y0_ODE_ut.pkl')
    save_data(y0L63, 'y0_ODE_L63.pkl')
    save_data(Lparamr, 'paramr_ODE_L63.pkl')
    save_data(gridt, 'grid_ODE_ut.pkl')
    save_data(lt, 'l_ODE_ut.pkl')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp  # changed from odeint to solve_ivp
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import pickle

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
    return interp1d(domain, grf_sample, kind='cubic', fill_value="extrapolate")

def poly_field(x,a):
    return np.sum([x**i*a[i] for i in range(len(a))],axis=0)

def exp_field(x,a):
    return np.sum([(np.exp(-x**2*i)-1)*a[i] for i in range(len(a))],axis=0)
def sin_field(x,a):
    return np.sum([np.sin(x*a[i]*np.pi)/len(a) for i in range(len(a))],axis=0)

def diffusion_pde(t, y, dx, u_interp):
    D = 0.01
    k = -0.1
    n = len(y)
    d2y_dx2 = np.zeros(n)
    d2y_dx2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
    u_values = u_interp(np.linspace(0, 1, n))
    du_dt = D * d2y_dx2 + k * y**2 + u_values
    return du_dt

def klein_gordon_pde(t, state, dx, u_interp):
    n = len(state) // 2
    u = state[:n]
    v = state[n:]
    
    d2u_dx2 = np.empty_like(u)
    
    # Interior points using central differences
    d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    
    # Toroidal (periodic) boundary conditions
    d2u_dx2[0] = (u[1] - 2*u[0] + u[-1]) / dx**2  # Left boundary
    d2u_dx2[-1] = (u[0] - 2*u[-1] + u[-2]) / dx**2  # Right boundary
    
    dudt = v
    dvdt = d2u_dx2 - u_interp(u)
    
    return np.concatenate([dudt, dvdt])

def GetSensors(U, sensor_locs):
    return U(sensor_locs).flatten()

def GenerateTimeseries_PDE(diffeq, y0, Trange, l, sensor_locs, x_grid, num_samples, save_path=None):
    def generate_sample():
        grf_sample = generate_grf(x_grid, l, num_samples=1)[0]
        u_interp = interpolate_grf(x_grid, grf_sample)
        t_random = np.random.uniform(Trange[0], Trange[1], Trange[2])
        T = np.sort(np.unique(np.concatenate(([0], t_random))))
        sol = solve_ivp(diffeq, (T[0], T[-1]), y0, t_eval=T, args=(x_grid[1] - x_grid[0], u_interp))
        sol = sol.y.T[1:]
        T = T[1:]
        sensor_vals = np.repeat(GetSensors(u_interp, sensor_locs).reshape(1,-1),Trange[2],axis=0)
        return T.reshape(-1, 1), sol, sensor_vals

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((np.vstack(ts), np.vstack(ys), np.vstack(us)), f)
    
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    # Check for rows containing nan, posinf, or neginf
    valid_ts_rows = ~np.any(np.isnan(ts_array) | np.isposinf(ts_array) | np.isneginf(ts_array), axis=1)
    valid_ys_rows = ~np.any(np.isnan(ys_array) | np.isposinf(ys_array) | np.isneginf(ys_array), axis=1)
    valid_us_rows = ~np.any(np.isnan(us_array) | np.isposinf(us_array) | np.isneginf(us_array), axis=1)
    valid_rows = valid_ts_rows & valid_ys_rows & valid_us_rows
    
    ts_array = ts_array[valid_rows]
    ys_array = ys_array[valid_rows]
    us_array = us_array[valid_rows]

    return ts_array, ys_array, us_array

def GenerateTimeseries_PDE_param(diffeq, y0, Trange, paramr,field, sensor_locs,x_grid, num_samples, save_path=None):
    def generate_sample():
        params =[np.random.uniform(p[0],p[1],1) for p in paramr]
        u_interp = lambda x:field(x,params)
        t_random = np.random.uniform(Trange[0], Trange[1], Trange[2])
        T = np.sort(np.unique(np.concatenate(([0], t_random))))
        sol_obj = solve_ivp(
            diffeq, 
            (T[0], T[-1]), 
            y0, 
            t_eval=T, 
            args=(x_grid[1] - x_grid[0], u_interp), 
            rtol=1e-6, 
            atol=1e-9, 
            dense_output=True
        )
        sol = sol_obj.sol(T).T[1:]
        T = T[1:]
        sensor_vals = GetSensors(u_interp, sensor_locs)
        # plt.scatter(sensor_locs,sensor_vals)
        # plt.show()
        return T.reshape(-1, 1), sol, np.repeat(sensor_vals.reshape(1,-1),Trange[2],axis=0)

    results = Parallel(n_jobs=-1)(delayed(generate_sample)() for _ in range(num_samples))
    ts, ys, us = zip(*results)
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump((np.vstack(ts), np.vstack(ys), np.vstack(us)), f)
    
    ts_array = np.vstack(ts)
    ys_array = np.vstack(ys)
    us_array = np.vstack(us)
    # Check for rows containing nan, posinf, or neginf
    valid_ts_rows = ~np.any(np.isnan(ts_array) | np.isposinf(ts_array) | np.isneginf(ts_array), axis=1)
    valid_ys_rows = ~np.any(np.isnan(ys_array) | np.isposinf(ys_array) | np.isneginf(ys_array), axis=1)
    valid_us_rows = ~np.any(np.isnan(us_array) | np.isposinf(us_array) | np.isneginf(us_array), axis=1)
    valid_rows = valid_ts_rows & valid_ys_rows & valid_us_rows
    
    ts_array = ts_array[valid_rows]
    ys_array = ys_array[valid_rows]
    us_array = us_array[valid_rows]

    return ts_array, ys_array, us_array

def y0_func(x,center,dx=1,dy=.1):
        return dy*np.exp(-(x-center)**2/(2*dx**2))

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)   

def open_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Spatial domain
    x_grid = np.linspace(0, 1, 100)

    # Sensor locations
    sensor_locs = np.random.uniform(0, 1, 300)

    x_grid_KG = np.linspace(0, 2, 200)

    # Sensor locations
    sensor_locs_KG = np.random.uniform(-3, 3, 300)

    y0= y0_func(x_grid,.75,.1)
    y0_KG= np.concat((y0_func(x_grid_KG[:len(x_grid_KG)//2],1/np.sqrt(2),.1,1),y0_func(x_grid_KG[len(x_grid_KG)//2:],1,.5,0)))

    # Time range
    Trange = [0, 1, 10]
    Trange_KG = [0, 3, 10]

    # Length-scale for GRF
    l = 0.1

    prange = [[1,3] for i in range(2)]

    # Number of samples for training and testing
    num_train_samples = 10000

    num_test_samples = 10000

    GenerateTimeseries_PDE(diffusion_pde, y0, Trange, l, sensor_locs, x_grid, num_train_samples, save_path='train_data_diff_PDE.pkl')
    GenerateTimeseries_PDE(diffusion_pde, y0, Trange, l, sensor_locs, x_grid, num_test_samples, save_path='test_data_diff_PDE.pkl')
    GenerateTimeseries_PDE(diffusion_pde, y0, [Trange[0],Trange[1],1000], l, sensor_locs, x_grid, 1, save_path='example_diff_PDE.pkl')


    GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, Trange_KG, prange,sin_field, sensor_locs_KG, x_grid_KG, num_train_samples, save_path='train_data_KG_PDE.pkl')
    GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, Trange_KG, prange,sin_field, sensor_locs_KG, x_grid_KG, num_test_samples, save_path='test_data_KG_PDE.pkl')
    GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, [Trange_KG[0],Trange_KG[1],1000], prange,sin_field, sensor_locs_KG, x_grid_KG, 1, save_path='example_KG_PDE.pkl')

    # tte_kg, yte_kg, ute_kg = GenerateTimeseries_PDE_param(klein_gordon_pde, y0_KG, [0,5,1000], prange,sin_field, sensor_locs_KG, x_grid_KG, 1, save_path='test_data_KG_PDE.pkl')
    # fig, axs = plt.subplots(1, 1)
    # X, T = np.meshgrid(x_grid_KG[:len(x_grid_KG)//2].reshape(-1, 1), tte_kg)

    # # Real Solution Heatmap
    # pc_real = axs.contourf(T, X, yte_kg[:,:len(x_grid_KG)//2], cmap='viridis', levels=100)    
    # cbar = fig.colorbar(pc_real, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

    # plt.show()
    save_data(x_grid, 'x_grid_PDE.pkl')
    save_data(sensor_locs, 'sensor_locs_PDE.pkl')
    save_data(Trange, 'Trange_PDE.pkl')
    save_data(y0, 'y0_PDE.pkl')

    save_data(x_grid_KG, 'x_grid_KG_PDE.pkl')
    save_data(sensor_locs_KG, 'sensor_locs_KG_PDE.pkl')
    save_data(Trange_KG, 'Trange_KG_PDE.pkl')
    save_data(y0_KG, 'y0_KG_PDE.pkl')
    save_data(prange, 'paramr_KG_PDE.pkl')
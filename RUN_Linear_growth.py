import os
try:
    os.chdir("Documents/Current_semester/Master")
except FileNotFoundError: pass
from KS_Compute import run_full_ETD
from KS_Compute import run_same_states
from KS_Compute import states
import numpy as np


if __name__ == '__main__': 

    L = 2*np.pi
    N = 2**9
    dt = 1/36000
    T = 0.005
    eq_str = 'np.random.normal(0, 10**(-2), size=len(x))' # np.random.normal(mu, sigma, size)
    
    ### Run one nu value ###
    nu = 0.0377
    include_modes = 10
    in_time = 0
    u_0 = eval(f"lambda x: {eq_str}")
    methods_to_run = ['ETD_RK4']
    run_full_ETD(nu, N, dt, T, u_0=u_0, L=L, plot=True, step_speed=1, save_name='Linear_growth', eq_str=eq_str, nest_folder=f'LINEAR_GROWTH/Single_nu_value/{nu}/{include_modes}/', mode='Linear_growth', start_mode=1, include_modes=include_modes, in_time=in_time)
    ### Run one nu value ###
    
    ### Run for all states ###
    for key, values in states.items():
        run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Linear_growth', start_mode=1, include_modes=10, in_time=0)
        run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Linear_growth', start_mode=1, include_modes=30, in_time=0)
    ### Run for all states ###
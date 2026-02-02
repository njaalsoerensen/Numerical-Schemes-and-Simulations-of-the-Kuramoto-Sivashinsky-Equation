import os
try:
    os.chdir("Documents/Current_semester/Master")
except FileNotFoundError: pass
from KS_Compute import solve_history
from KS_Compute import run_same_states
from KS_Compute import states
import numpy as np


if __name__ == '__main__': 
    
    eq_str = '-np.sin(x)'  # eq string to be able to save this info
    u_0 = eval(f"lambda x: {eq_str}")

    L = 2*np.pi
    N = 2**8
    freq = 2**10
    dt = 1/freq
    T = 15

    ### Run one nu value ###
    nu = 0.0377
    solve_history(nu, N, dt, T, u_0, L, save_name='Solve_history', nest_folder='Single_nu_value')
    ### Run one nu value ###
    
    # ### Run for all states ###
    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Solve_hostory', plot=True)
    # ### Run for all states ###
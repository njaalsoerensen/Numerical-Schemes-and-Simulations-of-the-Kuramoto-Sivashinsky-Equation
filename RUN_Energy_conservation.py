import os
try:
    os.chdir("Documents/Current_semester/Master")
except FileNotFoundError: pass
from KS_Compute import energy_confirmation
from KS_Compute import run_same_states
from KS_Compute import states
import numpy as np


if __name__ == '__main__': 
    
    eq_str = '-np.sin(x)'  # eq string to be able to save this info
    u_0 = eval(f"lambda x: {eq_str}")
    
    L = 2*np.pi
    N = 2**9
    freq = 2**9
    dt = 1/freq
    T = 25

    ### Run one nu value ###
    nu = 0.0377
    energy_confirmation(nu, N, dt, T, u_0, L=2*np.pi, in_time=5, plot=True, nest_folder='Single_nu_value', solve_method='ETD_RK4')
    ### Run one nu value ###
    
    ### Run all states ###
    for key, values in states.items():
        run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Confirm_energy', plot=True)
    ### Run for all states ###
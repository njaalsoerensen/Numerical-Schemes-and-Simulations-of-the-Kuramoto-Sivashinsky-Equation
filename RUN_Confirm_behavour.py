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
    N = 2**8
    dt = 1/220
    T = 30
    eq_str = '-np.sin(x)'
    u_0 = eval(f"lambda x: {eq_str}")
    
    ### Run one nu value ###
    # Note: Toggle the used solvers withub KS_Compute,run_full_ETD
    nu = 0.0377
    plot = False
    step_speed = 5
    mode = 'Confirm_behaviour'
    run_full_ETD(nu, N, dt, T, u_0, L=L, plot=plot, step_speed=step_speed, save_name='ETD_methods', eq_str=eq_str, nest_folder='/Confirm_behaviour/Single_nu_value', mode=mode)
    ### Run one nu value ###
    
    ### Run for all states ###
    for key, values in states.items():
        run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Confirm_behaviour', step_speed=5)
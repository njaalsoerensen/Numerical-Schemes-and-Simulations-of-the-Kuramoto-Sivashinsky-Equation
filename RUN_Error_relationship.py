import os
try:
    os.chdir("Documents/Current_semester/Master")
except FileNotFoundError: pass
from KS_Compute import error_relationship
import numpy as np


if __name__ == '__main__': 
    eq_str = '-np.sin(x)'
    u_0 = eval(f"lambda x: {eq_str}")
    L = 2*np.pi
    T = 30
    
    nums = [2**i for i in range(5, 10)]
    freqs = [2**i for i in range(5, 12)] # Means 'true' solutions will have freq[-1] frequency
    nums = nums[::-1]
    freqs = freqs[::-1]    
     
    
    solve_methods = ['ETD_RK4', 'ETD_RK4_CM', 'ETD_RK3', 'ETD_RK2']
    # solve_methods = ['ETD_RK4', 'ETD_RK4_CM']
    
    N = 2**8 # Means solutions for frequency variations will be run with N space points
    
    ### Run one nu value ###
    nu = 0.0377
    error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_methods=solve_methods)
    ### Run one nu value ###
    
    nu = 0.0245
    error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_methods=solve_methods)
    
    nu = 0.0355
    error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_methods=solve_methods)
    
    
    ### High frequiency computation for both order 4 methods ###
    nums = [2**i for i in range(3, 14)]
    freqs = [2**i for i in range(5, 16)] # Means 'true' solutions will have freq[-1] frequency
    nums = nums[::-1]
    freqs = freqs[::-1]
    N = 2**7
    
    # solve_methods = ['ETD_RK4', 'ETD_RK4_CM', 'ETD_RK3', 'ETD_RK2']
    solve_methods = ['ETD_RK4', 'ETD_RK4_CM']

    nu = 0.0245
    error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_methods=solve_methods, nest_folder='High_Accuracy')
    
    

        
    # From error relationship it seems like N = 2**6 is actually enough points for good convergence
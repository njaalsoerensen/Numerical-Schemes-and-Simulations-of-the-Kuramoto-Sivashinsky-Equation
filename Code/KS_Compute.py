import os
try:
    os.chdir("Documents/Current_semester/Prosjektoppgave")
except FileNotFoundError: pass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import KS_base
import KS_Schemes
import gc
import time
import Audun_solver_FS as Audun_solver_FS

def solve_history(nu, N, dt, T, u_0, L, save_name, nest_folder=False):
    solve_method = 'ETD_RK4'
    nu = round(nu, 4)
    
    folder = 'DATA/SOLVE_HISTORY/TIME/'
    if nest_folder:
        folder += f'/{nest_folder}'
    os.makedirs(folder, exist_ok=True)
        
    Bank = KS_base.DE_bank()

    Eq_ETD_RK4 = KS_Schemes.KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=False)  
    Eq_ETD_RK4.solve(method=solve_method)
    
    Bank.add_DE('Eq_ETD_RK4', Eq_ETD_RK4)
    Bank.plot_solve_history('Eq_ETD_RK4', title=f'{solve_method} solution, '+r'$\nu$ = '+f'{nu}', lab=0, step_speed=200, N_simul=1000, ani=False, save=True)

    name = f'Solve_history_{solve_method}_nu_{nu}_freq_{round(1/dt)}_N_{N}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    plt.show()

def error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_methods='ETD_RK4'):
    
    colors = {'ETD2': 'blue','ETD_RK2':'orange', 'ETD_RK3':'green', 'ETD_RK4':'red', 'ETD_RK4_CM':'purple', 'Audun_solver':'cyan'}
    
    
    # Plor theese in log log plots
    
    folder = 'DATA/ERROR_RELATIONSHIP/FREQUENCY'
    os.makedirs(folder, exist_ok=True)
    
    full_errors = []
    full_times = []
    Bank = KS_base.DE_bank()
    for method in solve_methods:  

        tot_times = []
        for freq in freqs:
            print(f'freq = {freq}')
            Bank.add_DE(f'Eq_{method}_{freq}', KS_Schemes.KS_FT(nu=nu, L=L, N=N, dt=1/freq, T=T, u_0=u_0, plot_initial=False))
            t_0 = time.time()
            Bank.bank[f'Eq_{method}_{freq}'].solve(method=method)
            tot_time = round(time.time() - t_0, 3)
            tot_times.append(tot_time)
        full_times.append(tot_times)
        full_errors.append([Bank.compute_error([f'Eq_ETD_RK4_{freqs[0]}', f'Eq_{method}_{freq}'], mode='freq') for freq in freqs])
        
    # Frequency error
    plt.title('Error vs frequency, '+r'$\nu$ = '+f'{nu}')
    plt.xlabel('Frequency')
    plt.ylabel('Total error')
    for i in range(len(solve_methods)):
        plt.plot(freqs[1:], full_errors[i][1:], label=solve_methods[i], color=colors[solve_methods[i]])
        plt.scatter(freqs[1:], full_errors[i][1:], color=colors[solve_methods[i]])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=1)
        
    name = f'Error_vs_frequency_nu_{nu}_N_{N}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    plt.show()
    
    # Freq Time spent computing
    plt.title('Runtime for Error vs frequency, '+r'$\nu$ = '+f'{nu}')
    plt.xlabel('Frequency')
    plt.ylabel('Time (s)')
    for i in range(len(solve_methods)):
        plt.plot(freqs, full_times[i], label=solve_methods[i], color=colors[solve_methods[i]])
        plt.scatter(freqs, full_times[i], color=colors[solve_methods[i]])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=2)
    
    name = f'RUNTIME_Error_vs_frequency_nu_{nu}_N_{N}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    plt.show()
    
    folder = 'DATA/ERROR_RELATIONSHIP/SPACE'
    os.makedirs(folder, exist_ok=True)
    
    freq = 2**10 # 1024
    full_errors = []
    full_times = []
    Bank = KS_base.DE_bank()   
    for method in solve_methods:
        tot_times = []
        for num in nums:
            print(f'N = {num}')
            Bank.add_DE(f'Eq_{method}_{num}', KS_Schemes.KS_FT(nu=nu, L=L, N=num, dt=1/freq, T=T, u_0=u_0, plot_initial=False))
            t_0 = time.time()
            Bank.bank[f'Eq_{method}_{num}'].solve(method=method)
            tot_time = round(time.time() - t_0, 3)
            tot_times.append(tot_time)
        full_times.append(tot_times)
        full_errors.append([Bank.compute_error([f'Eq_ETD_RK4_{nums[0]}', f'Eq_{method}_{num}'], mode='space', factor=round(nums[0]/num)) for num in nums])

    plt.title('Error vs space points, '+r'$\nu$ = '+f'{nu}')
    plt.xlabel('Space points (N)')
    plt.ylabel('Total error')
    for i in range(len(solve_methods)):
        plt.plot(nums[1:], full_errors[i][1:], color=colors[solve_methods[i]], label=solve_methods[i])
        plt.scatter(nums[1:], full_errors[i][1:], color=colors[solve_methods[i]])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=1)
    
    name = f'Error_vs_space_points_freq_{freq}_nu_{nu}_N_{N}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    
    plt.savefig(fname=f'Error_vs_space_points_freq_{freq}_nu_{nu}_N_{N}_T_{T}.png')
    plt.show()
    
    # space points Time spent computing
    plt.title('Runtime for Error vs space points, '+r'$\nu$ = '+f'{nu}')
    plt.xlabel('Space points')
    plt.ylabel('Time (s)')
    for i in range(len(solve_methods)):    
        plt.plot(nums, full_times[i], color=colors[solve_methods[i]], label=solve_methods[i])
        plt.scatter(nums, full_times[i], color=colors[solve_methods[i]])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=2)
    
    name = f'SPEED_Error_vs_space_points_nu_{nu}_N_{N}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    plt.show()
    
    
def energy_confirmation(nu, N, dt, T, u_0, L=2*np.pi, in_time=5, plot=False, nest_folder=False, solve_method='ETD_RK4'):
    folder = 'DATA/ENERGY_CONFIRMATION'
    os.makedirs(folder, exist_ok=True)    
    freq = int(1/dt)
    nu = round(nu, 4) # For ease of readabiilty and replication
    
    folder = 'DATA/ENERGY_CONFIRMATION/ENERGY_VS_TIME'
    if nest_folder:
        folder += f'/{nest_folder}'
    os.makedirs(folder, exist_ok=True)

    Eq_ETD_RK4 = KS_Schemes.KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=False)  
    Eq_ETD_RK4.solve(method=solve_method)
    plt.title(f'Energy vs time for {solve_method} scheme, '+r'$\nu$ = '+f'{nu}')
    plt.xlabel('Time (T)')
    plt.ylabel('Energy (E)')
    plt.plot(np.linspace(0, T, len(Eq_ETD_RK4.energy_log)), Eq_ETD_RK4.energy_log)    
    name = f'E_vs_t_nu_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    if plot:
        plt.show()
        
    folder = 'DATA/ENERGY_CONFIRMATION/Energy_conservation'
    if nest_folder:
        folder += f'/{nest_folder}'
    os.makedirs(folder, exist_ok=True)

    # np.diff calculates differential as D{y_i} = y_{i+1} - y_i, so need to divide by dt to get dy/dt    
    plt.title(f'Energy conservation equation {solve_method} scheme, '+r'$\nu$ = '+f'{nu}')
    plt.xlabel('Time (T)')
    plt.ylabel('Energy differential (dE)')
    plt.plot(np.linspace(0, T, len(Eq_ETD_RK4.energy_log)-1), np.diff(Eq_ETD_RK4.energy_log)/(dt), label='Energy differential LHS') # LHS
    plt.plot(np.linspace(0, T, len(Eq_ETD_RK4.energy_conservation_log)), Eq_ETD_RK4.energy_conservation_log, linestyle='--', label='Energy differential RHS') # RHS
    plt.legend()
    name = f'Energy_conservation_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    if plot:
        plt.show()
        
    folder = 'DATA/ENERGY_CONFIRMATION/E_vs_dE'
    if nest_folder:
        folder += f'/{nest_folder}'
    os.makedirs(folder, exist_ok=True)

    plt.xlabel('Energy (E)')
    plt.ylabel('Energy differential (dE)')        
    plt.title(f'Energy vs Energy differential {solve_method} scheme, '+r'$\nu$ = '+f'{nu}')
    plt.scatter(np.array(Eq_ETD_RK4.energy_log)[(freq*in_time)+1:][0], np.diff(Eq_ETD_RK4.energy_log)[(freq*in_time):][0]/(dt), label='Start')
    plt.plot(np.array(Eq_ETD_RK4.energy_log)[(freq*in_time)+1:], np.diff(Eq_ETD_RK4.energy_log)[(freq*in_time):]/(dt))
    plt.legend()
    name = f'E_vs_dE_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}.png'
    path = os.path.join(folder, name)    
    plt.savefig(fname=path)
    if plot:
        plt.show()


def run_full_ETD(nu, N, dt, T, u_0, L=2*np.pi, plot=False, step_speed=1, save_name='TEMP', eq_str=False, nest_folder=False, mode='', start_mode=0, include_modes=False, in_time=0):
    # # EDT these methods handle slow varying nonlinear term well
    print(f'\nrun_full_ETD, nu = {round(nu, 4)}')
    nu = round(nu, 4) # For reacreating pruposes I limit values 
    t_0 = time.time()
    
    '''
    Available modes
    -Confirm_behavior: Plots solutin in time domain and save animation
    -FT_energy: Plots solution in frequency domain as energy of modes and save animation
    -Confirm_energy: computes final energy of solution 'Energy should converge for this mode'
    -Linear_growth: Computes FT_energy but plots energy differential for all modes
    '''
    
    # Colors for consistent ploting
    colors = {'Eq_ETD2': 'blue','Eq_ETD_RK2':'orange', 'Eq_ETD_RK3':'green', 'Eq_ETD_RK4':'red', 'Eq_ETD_RK4_CM':'purple', 'Audun_solver':'cyan'}
    
    if mode=='Confirm_behaviour' or mode=='':
        # methods_to_run = ['Eq_ETD_RK2', 'Eq_ETD_RK3', 'Eq_ETD_RK4', 'Eq_ETD_RK4_CM']
        methods_to_run = ['Eq_ETD_RK4']
    if mode=='Confirm_energy' or mode=='FT_energy' or mode=='Linear_growth':
        methods_to_run = ['Eq_ETD_RK4'] 
        
    
        
        
    Bank = KS_base.DE_bank()
    # ETD1 seems unpredictably unstable, as a measure of accuracy this is expected to be the worst, fine to ignore
    for method_to_run in methods_to_run:
        Bank.add_DE(method_to_run, KS_Schemes.KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=False))
        Bank.bank[method_to_run].solve(method=method_to_run[3:])  
        if mode == 'FT_energy' or mode == 'Linear_growth': # Define important solutions as FT domai energyn, plots energy of modes
        
            Bank.bank[method_to_run].u = Bank.bank[method_to_run].u_hat
            Bank.bank[method_to_run].u = (Bank.bank[method_to_run].u*np.conjugate(Bank.bank[method_to_run].u)).real
            if include_modes:
                Bank.bank[method_to_run].u = Bank.bank[method_to_run].u[:,start_mode:start_mode+include_modes] # Incldues only the first modes   
            if in_time:
                in_time_f = int(len(Bank.bank[method_to_run].u)*(in_time/T))
                Bank.bank[method_to_run].u = Bank.bank[method_to_run].u[in_time_f:,:]
            if mode == 'Linear_growth':
                modes_energy = []
                modes_energy_diff = []
                for mode in Bank.bank[method_to_run].u:
                    # # Divinding my mode[0] gives weird stepwise behivour, probably due to machine precision?
                    # modes_energy.append(mode/mode[0])
                    # modes_energy_diff.append((np.diff(mode/mode[0])/dt))
                    modes_energy.append(mode)
                    modes_energy_diff.append((np.diff(mode)/dt))                    
                modes_energy_diff = np.array(modes_energy_diff).T
                modes_energy = np.array(modes_energy).T
                x_values = np.linspace(in_time, T, len(Bank.bank[method_to_run].u))
                x_values_diff = np.linspace(in_time, T, len(Bank.bank[method_to_run].u-1))

                # # Plot and save E
                # plt.title('Energy vs time for FT modes, '+r'$\nu$ = '+f'{nu}')
                # plt.xlabel('Time')
                # plt.ylabel('Energy (E)')  
                # for i in range(len(modes_energy)):
                #     plt.plot(x_values, modes_energy[i], label=f'Mode {start_mode+i}')
                # plt.legend(loc=1)
                
                # folder = 'DATA'
                # if nest_folder:
                #     folder += f'/{nest_folder}/E/'
                # os.makedirs(folder, exist_ok=True)
                
                # name = f'Linear_growth_Energy_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}_modes_{include_modes}.png'
                # path = os.path.join(folder, name)    
                # plt.savefig(fname=path)
                
                # plt.show()
                # # return                
                # # Plot and save dE
                # plt.title('Energy differential vs time for FT modes, '+r'$\nu$ = '+f'{nu}')
                # plt.xlabel('Time')
                # plt.ylabel('Energy differential (dE/dt)') 
                # for i in range(len(modes_energy_diff)):
                #     plt.plot(x_values_diff, modes_energy_diff[i], label=f'Mode {start_mode+i}')

                # plt.legend(loc=1)
                
                # folder = 'DATA'
                # if nest_folder:
                #     folder += f'/{nest_folder}/dE/'
                # os.makedirs(folder, exist_ok=True)
                
                # name = f'Linear_growth_dEnergy_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}_modes_{include_modes}.png'
                # path = os.path.join(folder, name)    
                # plt.savefig(fname=path)
                
                # plt.show()
                
                # Plot and save E log scale                
                plt.title('Energy vs time for FT modes, '+r'$\nu$ = '+f'{nu}')
                plt.xlabel('Time')
                plt.ylabel('Energy (E)')
                for i in range(len(modes_energy)):
                    plt.plot(x_values, modes_energy[i], label=f'Mode {start_mode+i}')
                plt.yscale('log')   
                plt.legend(loc=1)
                
                folder = 'DATA'
                if nest_folder:
                    folder += f'/{nest_folder}/E_logScale/'
                os.makedirs(folder, exist_ok=True)
                name = f'Linear_growth_Energy_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}_modes_{include_modes}.png'
                path = os.path.join(folder, name)    
                plt.savefig(fname=path)
                plt.show()  
                
                # Save X-O values for this case
                total_num = round(T/dt)
                print(total_num)
                num = int(0.1*total_num)
                modes = np.array([start_mode + i for i in range(len(modes_energy))])
                X_y = np.array([np.log(modes_energy[i][num]/modes_energy[i][0])/(num*dt)  for i in range(len(modes_energy))])
                
                O_y = modes**2 - Bank.bank[method_to_run].nu*modes**4
                
                plt.title('Computed vs True linear growth rate for FT modes,' +r'$\nu$ = '+f'{nu}')
                plt.xlabel('Mode')
                plt.ylabel('Growth rate')
                plt.scatter(modes, O_y, marker='o', label='True linear growth rate')
                plt.scatter(modes, X_y, marker='x', label='Computed linear growth rate')
                plt.legend(loc='lower left')                
                folder = 'DATA'
                if nest_folder:
                    folder += f'/{nest_folder}/LGR_low_num/'
                os.makedirs(folder, exist_ok=True)
                name = f'Liner_growth_rate_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}_modes_{include_modes}_num_{num}.png'
                path = os.path.join(folder, name)    
                plt.savefig(fname=path)
                plt.show()  
                
                # Higher_num
                num = int(0.9*total_num)
                X_y = np.array([np.log(modes_energy[i][num]/modes_energy[i][0])/(num*dt)  for i in range(len(modes_energy))])
                
                
                plt.title('Computed vs True linear growth rate for FT modes,' +r'$\nu$ = '+f'{nu}')
                plt.xlabel('Mode')
                plt.ylabel('Growth rate')
                plt.scatter(modes, O_y, marker='o', label='True linear growth rate')
                plt.scatter(modes, X_y, marker='x', label='Computed linear growth rate')
                plt.legend(loc='lower left')               
                folder = 'DATA'
                if nest_folder:
                    folder += f'/{nest_folder}/LGR_large_num/'
                os.makedirs(folder, exist_ok=True)
                name = f'Liner_growth_rate_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}_modes_{include_modes}_num_{num}.png'
                path = os.path.join(folder, name)    
                plt.savefig(fname=path)
                plt.show()  
          
                # # Plot and save dE
                # plt.title('Energy differential vs time for FT modes, '+r'$\nu$ = '+f'{nu}')
                # plt.xlabel('Time')
                # plt.ylabel('Energy differential (dE/dt)')
                # for i in range(len(modes_energy_diff)):
                #     plt.plot(x_values_diff, modes_energy_diff[i], label=f'Mode {start_mode+i}')
                # plt.yscale('log')   
                # plt.legend(loc=1)
                
                # folder = 'DATA'
                # if nest_folder:
                #     folder += f'/{nest_folder}/dE_logScale/'
                # os.makedirs(folder, exist_ok=True)
                
                # name = f'Linear_growth_dEnergy_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_inTime_{in_time}_modes_{include_modes}.png'
                # path = os.path.join(folder, name)    
                # plt.savefig(fname=path)
                
                # plt.show()
                return                  

        if mode == 'Confirm_energy':
            return  Bank.bank[method_to_run].energy_log[-1]

    # Setup series
    fig, ax = plt.subplots()
    x_values = Bank.bank[methods_to_run[0]].x_values
    series = [ax.plot(x_values, Bank.bank[method].u[0], label=method[3:], color=colors[method])[0] for method in methods_to_run]
    # ax.legend(loc=1) # Upper right
    if mode == 'Confirm_energy':
        ax.set_title(r'ETD methods $\nu$ = ' + f'{round(nu, 4)}')
        ax.legend(loc=1) # Upper right  
    if mode == 'FT_energy':
        method = methods_to_run[0]
        series.append(ax.scatter(x_values, Bank.bank[method].u[0], color=colors[method]))
        
    # For Audun solver
    KS_FS_sin = Audun_solver_FS.different_initial_data(nu=nu, max_wavenumber=int(N/2), tmax=T, dt=dt)  
    # KS_FS_sin.u_hat = KS_FS_sin.u_fourier.T # Actaully u_hat
    KS_FS_sin.u = KS_FS_sin.u.T
    print(KS_FS_sin.u.shape)
    print(Bank.bank['Eq_ETD_RK4'].u.shape)
    Bank.add_DE('Audun_solver', KS_FS_sin)    
    series.append(ax.plot(x_values, Bank.bank['Audun_solver'].u[0], label='Audun_solver', color=colors['Audun_solver'])[0])
    print('this is mode', mode)
    methods_to_run.append('Audun_solver')
    ax.legend(loc=1) # Upper right
    ax.set_title(r'ETD methods $\nu$ = ' + f'{round(nu, 4)}')
        
    # Animation of series
    def animate(t): # Returns time domain solution
        for i in range(len(methods_to_run)):
            if methods_to_run != 'Audun_solver':
                Bank.find_plot_lims(Bank.bank[methods_to_run[i]], t*step_speed)                    
            series[i].set_ydata(Bank.bank[methods_to_run[i]].u[t*step_speed])
        plt.ylim((Bank.min_y*1.1, Bank.max_y*1.1))
        if mode=='Confirm_behaviour':
            ax.set_title(r'Solver methods $\nu$ = ' + f'{round(nu, 4)}, Eq: {eq_str}, t = {round(t*step_speed*dt, 2)}')
        # if mode=='FT_energy':
        #     ax.set_title(f'{methods_to_run[0][3:]} mode energy ' + r' $\nu$ = ' + f'{round(nu, 4)}, Eq: {eq_str}, t = {round(t*step_speed*dt, 2)}')
        #     method = methods_to_run[0]
        #     series[1].set_offsets(np.c_[x_values, Bank.bank[method].u[t*step_speed]])
        return series

    # Framerate and run animation
    framerate = 30
    print('-Animation creating')
    ani = animation.FuncAnimation(fig, 
                                  animate, 
                                  frames=int(Bank.bank[methods_to_run[0]].u.shape[0]/step_speed), 
                                  interval=int(1000/framerate), 
                                  blit=True, 
                                  repeat=False)
    # Save animation  
    print('-Animation saving')
    folder = 'DATA/ANIMATION/GIF'
    if nest_folder:
        folder = f'DATA/GIF/{nest_folder}'
    os.makedirs(folder, exist_ok=True)
    save_name += f'_nu_{nu}_N_{N}_freq_{int(1/dt)}_T_{T}_stepSpeed_{step_speed}'  
    path = os.path.join(folder, save_name)
    ani.save(path+'.gif', writer='pillow', fps=framerate)
    
    # Saving npy
    print('-Numpy saving')
    folder = 'DATA/ANIMATION/NPY'
    if nest_folder:
        folder = f'DATA/NPY/{nest_folder}'
    os.makedirs(folder, exist_ok=True)
    for method in methods_to_run:
        name = method[3:]+'_'+save_name+'.npy'
        path = os.path.join(folder, name)
        np.save(path, Bank.bank[method].u)
    total_time = time.time() - t_0
    print(f'Finished in {round(total_time,2)} seconds')
    
    
def run_same_states(N, dt, T, u_0, L, eq_str, state_type, nu_values, num=5, mode='', step_speed=1, plot=False, nest_folder=False,start_mode=0, include_modes=20, in_time=0):
    # Compues for all same states
    print(f'\n{state_type}')
    u_0 = eval(f"lambda x: {eq_str}") 
    
    for i in range(len(nu_values)):
        nu_linspace = np.linspace(nu_values[i][0], nu_values[i][1], num+2)[1:-1] # Run for interior points to avoid boundary behaviour
        for nu in nu_linspace:
            if mode == 'Confirm_behaviour':                
                run_full_ETD(nu, N, dt, T, u_0, L=L, plot=plot, step_speed=step_speed, save_name='ETD_methods', eq_str=eq_str, nest_folder='Confirm_behaviour/'+state_type+f'_{i}', mode=mode)
                
            if mode == 'FT_energy':
                run_full_ETD(nu, N, dt, T, u_0, L=L, plot=plot, step_speed=step_speed, save_name='FT_energy', eq_str=eq_str, nest_folder='FT_energy/'+state_type+f'_{i}', mode=mode, include_modes=include_modes)
                
            if mode == 'Linear_growth':    
                run_full_ETD(nu, N, dt, T, u_0, L=L, plot=plot, step_speed=step_speed, save_name='Linear_growth', eq_str=eq_str, nest_folder='Linear_growth/'+state_type+f'_{i}', mode=mode, start_mode=start_mode, include_modes=include_modes, in_time=in_time)
            
            if mode == 'Confirm_energy':
                eq_str = eq_str.replace('.', '_')
                energy_confirmation(nu, N, dt, T, u_0, L, in_time=5, plot=plot, nest_folder=eq_str+'/'+state_type+f'_{i}')    
            
            if mode == 'Solve_hostory':
                eq_str = eq_str.replace('.', '_')
                solve_history(nu, N, dt, T, u_0, L, save_name='Solve_hostory', nest_folder=eq_str+'/'+state_type+f'_{i}')

    
if __name__ == '__main__':
    gc.collect() 
    
    Constant_states = [[1, 9]] # From 1 to infty
    
    Fully_modeal_steady_attractors = [[0.2475, 1],
                                     [0.66697, 0.0755],
                                     [0.03965, 0.055235]]

    Bimodal_steady_attractors = [[0.0756, 0.2472]]

    Trimodeal_steady_attractors = [[0.0599, 0.06695]]

    Tetramodal_steady_attractors = [[0.0344, 0.037348]]

    Periodic_attractors = [[0.055238, 0.05985],
                           [0.03735, 0.0396]]

    Periodic_attractors_complete_period_doubling = [[0.029756, 0.0343],
                                                    [0.024, 0.0251]]

    Chaotic_oscillations = [[0.0252, 0.029755],
                            [0.01, 0.023]] # Lower bound for this is not actually known
    
    # # Total of 12 ranges for behaviour 
    states = {'Periodic_attractors_complete_period_doubling': Periodic_attractors_complete_period_doubling,
              'Constant_states': Constant_states,
              'Fully_modeal_steady_attractors': Fully_modeal_steady_attractors,
              'Bimodal_steady_attractors': Bimodal_steady_attractors,
              'Trimodeal_steady_attractors': Trimodeal_steady_attractors,
              'Tetramodal_steady_attractors': Tetramodal_steady_attractors,
              'Periodic_attractors': Periodic_attractors, 
              'Chaotic_oscillations': Chaotic_oscillations} 
    
    # states = {'Chaotic_oscillations': Chaotic_oscillations}
    
    # states = {'Periodic_attractors_complete_period_doubling': Periodic_attractors_complete_period_doubling}
    
    # # <Solve history>
    # eq_str = '-np.sin(x)'  # eq string to be able to save this info
    # u_0 = eval(f"lambda x: {eq_str}")
    
    # L = 2*np.pi
    # N = 2**8
    # freq = 2**10
    # dt = 1/freq
    # T = 15

    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Solve_hostory', plot=True)
    # # </Solve history>        
    
    
    
    # # <Energy conservation conformation>
    # eq_str = '-np.sin(x)'  # eq string to be able to save this info
    # u_0 = eval(f"lambda x: {eq_str}")
    
    # L = 2*np.pi
    # N = 2**8
    # freq = 2**11
    # dt = 1/freq
    # T = 25
    
    # nu = 0.057
    
    # energy_confirmation(nu, N, dt, T, u_0, L, in_time=5, plot=True, nest_folder='something') 

    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Confirm_energy', plot=True)
    # # </Energy conservation confirmation>    
    
    
    # # <Confirm behaviour>
    # L = 2*np.pi
    # N = 2**8
    # dt = 1/220
    # T = 15
    # # eq_str = 'np.cos(x)' # eq string to be able to save this info
    # eq_str = 'np.cos(2*np.pi*x/L)'
    # u_0 = eval(f"lambda x: {eq_str}")
    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Confirm_behaviour', step_speed=5)
    # # </Confirm behaviour>
    
    
    
    # # <Confirm behaviour FT>
    # L = 2*np.pi
    # N = 2**8
    # dt = 1/220
    # T = 15
    # # eq_str = 'np.cos(x)' # eq string to be able to save this info
    # eq_str = '-np.sin(x)'
    # u_0 = eval(f"lambda x: {eq_str}")
    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='FT_energy', step_speed=2)
    # # </Confirm behaviour FT>
    
    # <Confirm behaviour FT_FS>
    L = 2*np.pi
    N = 2**8
    dt = 1/2**10
    T = 15
    # eq_str = 'np.cos(x)' # eq string to be able to save this info
    eq_str = '-np.sin(x)'
    u_0 = eval(f"lambda x: {eq_str}")
    for key, values in states.items():
        run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Confirm_behaviour', step_speed=20, num=2)
    # </Confirm behaviour FT>
    
    # # <Linear growth>
    # L = 2*np.pi
    # N = 2**8
    # dt = 1/36000
    # T = 0.005
    # # eq_str = '-np.sin(x)' # eq string to be able to save this info
    # eq_str = 'np.random.normal(0, 10**(-2), size=len(x))' # np.random.normal(mu, sigma, size)
    
    # u_0 = eval(f"lambda x: {eq_str}")
    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Linear_growth', start_mode=1, include_modes=30, in_time=0)
    # # </Liner growth>
    

    # <Testing vectorized version>
    # eq_str = 'np.cos(2*np.pi*x/L)' # eq string to be able to save this info
    # u_0 = eval(f"lambda x: {eq_str}")
        
    # run_full_ETD(nu, N, dt, T, u_0, L=L, plot=True, step_speed=3, save_name='ETD_methods', eq_str=eq_str, mode='Confirm_behaviour') 
    # # </Testing vectorized version>
    
    # <Error relatioonship>
    # Timesteps for simulation
    # 3600 = 2^4 3^2 5^2 / 1
    # 1800 = 2^3 3^2 5^2 / 2
    # 1200 = 2^4 3^1 5^2 / 3
    # 900  = 2^2 3^2 5^2 / 4
    # 720  = 2^4 3^2 5^1 / 5
    # 600  = 2^3 3^1 5^2 / 6
    # 450  = 2^1 3^2 5^2 / 8
    # 400  = 2^4 3^0 5^2 / 9
    # 360  = 2^3 3^2 5^1 / 10
    # 300  = 2^2 3^1 5^2 / 12
    # 240  = 2^4 3^1 5^1 / 15
    # 225  = 2^0 3^2 5^2 / 16
    # 200  = 2^3 3^1 5^1 / 18
    # 180  = 2^2 3^2 5^1 / 20
    # 150  = 2^1 3^1 5^2 / 24
    
    # Kan gold velge 2^x for punkter of frekvenser
    
    # eq_str = 'np.cos(2*np.pi*x/L)' # eq string to be able to save this info
    # eq_str = '-np.sin(x)'
    # u_0 = eval(f"lambda x: {eq_str}")
    
    # solve_methods = ['ETD_RK4', 'ETD_RK4_CM', 'ETD_RK3', 'ETD_RK2']
    
    # # freqs = np.array([3600, 1800, 1200, 900, 720, 600, 450, 400, 360, 300, 240, 200, 180, 150, 100, 50, 30, 10])
    # nums = [2**i for i in range(3, 10)]
    # nums = nums[::-1]
    
    # freqs = [2**i for i in range(5, 13)]
    # freqs = freqs[::-1]
    
    # L = 2*np.pi
    # N = 2**8
    # # nu = 0.377 # Periodic attractor, intersting behavour
    # # nu = 0.0245 # Other periodic attractor
    # nu = 0.0355 # Stationary solution
    # Bank = KS_base.DE_bank()
    # # error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_method=metod)
    # T = 30
    # error_relationship(eq_str, u_0, L, N, nu, T, freqs, nums, solve_methods=solve_methods)

        
    # From error relationship it seems like N = 2**6 is actually enough points for good convergence
    
    # </Error relationship>
        
    
    # # <Average energy>
    # L = 2*np.pi
    # N = 2**7
    # dt = 1/300
    # T = 20
    # eq_str = 'np.cos(2*np.pi*x/L)' # eq string to be able to save this info
    # u_0 = eval(f"lambda x: {eq_str}")
    # for key, values in states.items():
    #     run_same_states(N, dt, T, u_0, L, eq_str, key, values, mode='Confirm_energy', num=15)
    # # </Average energy>
    
    

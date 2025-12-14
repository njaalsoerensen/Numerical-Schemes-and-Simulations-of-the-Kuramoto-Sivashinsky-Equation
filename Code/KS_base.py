import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class DE_bank:
    # Handles several KS_equations
    def __init__(self):
        self.bank = {}
        self.min_y, self.max_y = 0, 0
    
    def add_DE(self, diff_name, diff):
        '''
        Parameters
        ----------
        diff : KS_equation
        diff_name : STR
            Name of KS equation
        '''
        self.bank[diff_name] = diff
        
    def find_plot_lims(self, Eq, t='blank', mode='time'):
        if t == 'blank':
            t = Eq.i_index
        if mode=='time':
            y_values = Eq.u[t]
        if mode=='fourier':
            y_values = Eq.u_hat[t]
        if max(y_values) > self.max_y:
            self.max_y = max(y_values)
        if min(y_values) < self.min_y:
            self.min_y = min(y_values)
            
    def plot_solve(self, diff_names, labs=False, title=0, step_speed=1, mode='time', show=True):
        '''
        Parameters
        ----------
        diff_name : STR
            Name of KS_equation.
        tit : STR, optional
            Title of plot. The default is 0.
        lab : Bool, optional
            Wheter to use labeles. The default is False.

        Returns
        -------
        None.

        '''
        
        ## ADD DIFFERENT PLOT MODE FOR FOURIER MODE
        
        Eq_s = [self.bank[diff_name] for diff_name in diff_names]
        T, N = Eq_s[0].u.shape
        for i in range(len(Eq_s)):
            assert T == Eq_s[i].u.shape[0], 'Equations have different Time_dim'
            assert N == Eq_s[i].u.shape[1], 'Equations have different Space_dim' # In some siutations this comparison could also be interesting
        self.min_y, self.max_y = 0, 0
        print('Ploting solutions in ', mode, 'domain')
        plt.figure()
        for t in tqdm(range(0, T, step_speed)):
            for i in range(len(Eq_s)):
                Eq = Eq_s[i]
                if mode == 'time':
                    plt.xlabel('x values')
                    plt.ylabel('y values')
                    self.find_plot_lims(Eq, t, mode=mode)
                    if labs:
                        plt.plot(Eq.x_values, Eq.u[t], label=diff_names[i])
                        plt.legend(loc='upper right')
                    else: plt.plot(Eq.x_values, Eq.u[t], label=diff_names[i])
                if mode == 'fourier':
                    plt.xlabel('Mode')
                    plt.ylabel('Amplitude')
                    self.find_plot_lims(Eq, t, mode=mode)
                    if labs:
                        plt.plot(np.arange(round(Eq.u.shape[1])/2+1), Eq.u_hat[t], label='FT{'+diff_names[i]+'}')
                        plt.legend(loc='upper right')
                    else: plt.plot(np.arange(round(Eq.u.shape[0]/2+1)), Eq.u_hat[t], label='FT{'+diff_names[i]+'}')                  
            if title:
                plt.title(title)
            plt.ylim((self.min_y*1.1, self.max_y*1.1))
            if show:
                plt.show()
                
    def plot_solve_history(self, diff_name, title=0, lab=0, step_speed=1, N_simul=100, ani=False, save=0):
        '''
        Parameters
        ----------
        diff_name : Str
            Name of KS_equation.
        title : Str, optional
            Title of plot. The default is 0.
        lab : STR, optional
            Label for plot. The default is 0.
        step_speed : Int, optional
            Timestep intervall. The default is 1.
        N_simul : Int, optional
            Number of simultainus solution in plot. The default is 100.

        Returns
        -------
        None.
        '''
        # For a solved equations goes through entire solve history as contur plot
        Eq = self.bank[diff_name] 
        t_values = np.linspace(0, Eq.t, Eq.u.shape[0]) 
        plt.contourf(t_values, Eq.x_values, Eq.u.T, cmap='plasma')
        plt.xlabel('time')
        plt.ylabel('x values')
        if title:
            plt.title(title)
        if lab:
            plt.legend()
        plt.colorbar()
        if save:
            return
        plt.show()
        if not ani:
            return
        for i in tqdm(range(0, Eq.u.shape[0]-N_simul, step_speed)):
            plt.contourf(t_values[i:i+N_simul], Eq.x_values, Eq.u[i:i+N_simul].T, cmap='plasma')
            plt.xlabel('time')
            plt.ylabel('x values')
            if title:
                plt.title(title)
            if lab:
                plt.legend()
            plt.colorbar()
            plt.show()  
            
    def compute_error(self, diff_names, mode='freq', factor=1):
        Eq_s = [self.bank[diff_name] for diff_name in diff_names]
        T, N = Eq_s[0].u.shape
        TS = [Eq_s[i].u[-1] for i in range(2)] # Time series of last solutino, each entry is an array  
        if mode == 'freq':      
            MSE = (1/N)*np.sum((TS[1]-TS[0])**2)
        if mode == 'space': 
            MSE = (1/Eq_s[1].u.shape[1])*np.sum((TS[1]-TS[0][::factor])**2)    
        return MSE
            

class KS_equation:
    # General KS_equation with method for discretization
    def __init__(self, nu, L, N, dt, T, u_0, plot_initial=False):
        # Features of Eq
        self.nu = nu 
        # Discretization grid
        self.h = dt
        self.L = L
        self.k = L/N
        self.x_values = np.arange(N)*L/N       
        self.u = np.zeros((int(T/dt), N))
        # Initial function
        self.u[0] = u_0(self.x_values) # intital function a t_0
        # Stability condition        
         # print(self.h/self.k**4) # Stable value is 1.0764662608687885
        # Indexing
        self.t = 0
        self.i_index = 0
        # Plotting
        self.max_y, self.min_y = 0, 0
        if plot_initial: self.plot_initial(title='Initial curve')
        
    def solve(self, method):
        method_to_run = getattr(self, method)
        print('-Solving ', method)
        for _ in tqdm(range(self.u.shape[0]-1)):
        # for _ in range(self.u.shape[0]-1):
            method_to_run()
        
    def timestep(self):
        self.i_index += 1      
        self.t = self.i_index*self.h
        
    def append_solve(self, arr):
        # self.solve_history = np.vstack((self.solve_history, arr))
        self.timestep()
        self.u[self.i_index] = arr
        
    def plot_initial(self, title=0, lab=0):
        y_values = self.u[self.i_index]
        plt.xlabel('x values')
        plt.ylabel('y values')
        # min_y, max_y = self.find_plot_lims()
        # plt.ylim((min_y, max_y))
        if lab:
            plt.plot(self.x_values, y_values, lab=lab)
            plt.legend()
        else:
            plt.plot(self.x_values, y_values)
        if title:
            plt.title(title)
        plt.show()
    
    def reduce_shape(self, T=False, frac=False, revert_change=False):
        # Reduce time solutions to int T solutions, or by a fraction frac<1
        v = self.u
        self.i_index = 0
        assert T != frac, 'T AND frac CAN NOT BOTH BE ACTIVE'
        if T: # Idea is to use interpolation
            self.u = np.zeros((int(T*v.shape[0]/self.h), v.shape[1]))
            for i in tqdm(range(0, T, int(v.shape[0]/T))):
                self.append_solve(v[i])
        if frac:
            self.u = np.zeros((int(v.shape[0]*frac), v.shape[1]))
            self.u[0] = v[0]
            self.h *= round(1/frac)
            for i in range(1, self.u.shape[0]-1):
                self.append_solve(v[round(i/frac)])
            self.u[-1] = v[-1]
        
        

if __name__ == '__main__':
    print('Hello world')
    
    
# Written by Audun Theodorsen
import numpy as np
import kuramoto_sivashinsky_helper as ksh

class KSsystem:
    """
    Solve the KS equation:
    d_t u(x,t) + d_xx u(x,t) + nu d_xxxx u(x,t) + u(x,t) d_x u(x,t) = 0
    u(x+2pi,t) = u(x,t)
    u(x,0) = init(x)
    using a fourier expansion.
    
    Input:
        nu: hyperviscosity parameter
        init: initial profile. If None, uses a sinusoidal perturbation with the first wavenumber, u(x,0) = -sin(x)
        max_wavenumber: Largest wavenumber, corresponds to the nyquist frequency

    Methods:
        run_KS(T, dt): Runs the simulation for a time T with time step dt.
                       Saves the output of solve_ivp as 
                       self.t : time points of the evaluation and
                       self.u_fourier : the solution in Fourier space as a (max_wavenumber, self.t.size) numpy array.

        calculate_energy(): Saves u(t)^2/2 in the KS as a function of time as self.energy

        calculate_spatial_solution(): From the IFT of self.res as
                        self.x: spatial array 2pi*[0,M]/M with M = 2*max_wavenumber
                        self.u: solution of KS as an (x.size, t.size) - numpy array

        calculate_wavenumber_spectrum(cutoff=0):  Save <u_k u_{-k}> where the average is over time as self.spectrum
                                                 cutoff is a cutoff in time to remove the initial transient.

    """
    def __init__(self, nu, max_wavenumber, init='sin'):
        self.nu = nu
        self.max_wavenumber = max_wavenumber

        if type(init) in [list, np.ndarray]:
            assert(len(init)==max_wavenumber)
            self.init = np.array(init, dtype = 'complex')
        elif init == 'sin':
            self.init = np.zeros(max_wavenumber,dtype = 'complex')
            self.init[0] = 0.5j # Corresponds to -sin(x).
        elif init == 'random':
            self.init = np.random.laplace(scale=1,size=max_wavenumber) + 1.j*np.random.laplace(scale=1,size=max_wavenumber)
        elif init == 'random_sine':
            # Only imaginary numbers activated
            self.init = 1.j*np.random.laplace(scale=1,size=max_wavenumber)
        else:
            print('Invalid init')

    def run(self, T, dt):
        self.res = ksh.run_KS_finite_wavenumber(self.nu, self.max_wavenumber, self.init, T, dt)
        self.t = self.res.t # Just renaming - references the same object
        self.u_fourier = self.res.y # Just renaming - references the same object

    def _run_sine(self, T, dt):
        # Just the sine transform
        self.res = ksh.run_KS_finite_wavenumber_sine(self.nu, self.max_wavenumber, np.imag(self.init), T, dt)
        self.t = self.res.t # Just renaming - references the same object
        self.u_fourier = 1.j*self.res.y # This is just the imaginary component

    def calc_energy(self):
        # See discussion in init - we calculate the energy as u_fourier*conjugate(u_fourier), so 
        # we must compensate by dx for each u_fourier to get the true energy.
        #dx = 0.5/self.max_wavenumber
        self.energy = np.real(np.sum(self.u_fourier*np.conj(self.u_fourier),axis=0))#*dx**2

    def calc_spatial_solution(self):
        # The zero frequency is zero for all time, but we need it for the correct number of waves
        W = np.concatenate((np.zeros((self.t.size,1)).T,self.u_fourier), axis=0)
        self.u = np.fft.irfft(W,norm='forward',axis=0)
        
        M = 2*self.max_wavenumber
        self.x=2*np.pi*np.arange(0,M)/M

    def calc_wavenumber_spectrum(self, cutoff=0):
        self.spectrum = ksh.wavenumber_spectrum(self.res, cutoff)        

    def calc_aliasing_error(self):
        self.aliasing_error = ksh.test_dealias(self.res)
    

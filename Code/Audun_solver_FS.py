# Most written by Audun Theodorsen and sligtly changed by Njål Gunnar Sørensen
import os
try:
    os.chdir("Documents/Current_semester/Prosjektoppgave")
except FileNotFoundError: pass


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import kuramoto_sivashinsky_sim as ks

def run_and_plot_KS(nu, max_wavenumber, tmax, dt, init='random'):
    KS = ks.KSsystem(nu, max_wavenumber, init=init)
    KS.run(tmax, dt) # Time and fourier solution in KS.t, Ks.u_fourier
    KS.calc_energy() # Energy stored in KS.energy
    KS.calc_spatial_solution() # Spatial solution stored as KS.x, KS.u
    #KS.calc_aliasing_error()


    #plt.figure()
    #plt.plot(KS.t, KS.aliasing_error)
    
    plt.figure('Energy')
    plt.title('Energy')
    plt.plot(KS.t, KS.energy)
    plt.xlabel('t')
    plt.ylabel('E')
    plt.ylim(0,100)
    #plt.savefig(f'energy_nu_{nu}_'+init+'.png')

    plt.figure('Energy vs energy derivative')
    plt.title('Energy vs energy derivative')
    # If the attractor is periodic, the energy should be as well. 
    plt.plot(KS.energy[KS.t>20][:-1], np.diff(KS.energy[KS.t>20])/dt)
    plt.xlabel('E')
    plt.ylabel('dE/dt')

    plt.figure('Abs value of each wavenumber final')
    plt.title('Abs value of each wavenumber final')
    # Plotting the absolute value of each wavenumber gives a quick check of which wavenumbers are activated,
    # is the attractor bimodal, trimodal etc?
    plt.semilogy(np.arange(1, max_wavenumber+1), np.abs(KS.u_fourier[:,-1]),'o')

    plt.figure('Power spectral density')
    plt.title('PSD')
    ft = np.fft.rfft(KS.u[int(KS.x.size/4),int(10/dt):])
    PSD = np.real(ft*np.conj(ft))
    plt.semilogy(PSD)
    
    plt.figure('Compare init and final')
    plt.title('Comparison of initial and final states')
    plt.plot(KS.x, KS.u[:,0], label='u init')
    plt.plot(KS.x, KS.u[:,-1], label='u final')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()


    # In the next two plots we make 3d and contour plots. For these, we need x and t - values in a grid
    # since u is in a grid. The way to do this for evenly spaced data is to use meshgrid.
    # Note that the order depends on the indexing. The two lines below give the same result and
    # are consistent with KS.u which has KS.u.shape=(KS.x.size, KS.t.size)
    
    T, X = np.meshgrid(KS.t,KS.x, indexing='xy')
    #X, T = np.meshgrid(KS.x,KS.t, indexing='ij')
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    tcut = int(5/dt)#int(tmax*0.25) # We don't want to look at the transient
    ax.plot_surface(T[:,tcut:], X[:,tcut:], KS.u[:,tcut:], cmap=cm.cividis)
    #ax.set_zlim([-0.05, 0.05]) # Another way to zoom in on the solution
    plt.title('u(x,t)')
    plt.xlabel('t')
    plt.ylabel('x')

    plt.figure(figsize = (10,50))
    ax = plt.subplot(1,1,1)
    # Specify the countours in contourf to see the solution better - could also cut as above.
    # levels = np.linspace(np.min(KS.u[:,-1]),np.max(KS.u[:,-1]),5)
    plt.contourf(T[:,tcut:], X[:,tcut:], KS.u[:,tcut:],levels=5,cmap=cm.cividis)
    ax.set_aspect('equal')
    plt.title('u(x,t) contours')
    plt.xlabel('t')
    plt.ylabel('x')
    #plt.savefig(f'contours_nu_{nu}_'+init+'.png', dpi=300)


def different_initial_data(nu, max_wavenumber, tmax, dt):
    # KSsin = ks.KSsystem(nu, max_wavenumber, init='-sin') # u(x,0) = -sin(x) ?
    
    KSsin = ks.KSsystem(nu, max_wavenumber, init='sin') # u(x,0) = sin(x)
    KSsin.run(tmax, dt)
    
    
    KSsin.calc_spatial_solution() # To get .u
    
    return KSsin
    # KSrand = ks.KSsystem(nu, max_wavenumber, init='random') # Each u_k is a Laplace distributed random number in both real and complex values

    # # Custom initial data
    # init = np.zeros(max_wavenumber, dtype='complex')
    # init[1] = 0.1j*max_wavenumber
    # init[2] = 5j*2*max_wavenumber
    # KScustom = ks.KSsystem(nu, max_wavenumber, init)

    


    # #plt.figure()
    # for KS, label, col in zip([KSsin, KSrand, KScustom],
    #                           ['sin','random','custom'],
    #                           ['C0','C1','C2']):
    #     KS.run(tmax, dt)
    #     KS.calc_energy()
    #     plt.semilogy(KS.t, KS.energy, label=label)
    #     plt.plot()
    # plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('E')

    # for KS, label in zip([KSsin, KSrand, KScustom],
    #                      ['sin','random','custom']):
    #     KS.calc_spatial_solution()
    #     plt.figure('u contour' + label)
    #     ax = plt.subplot(1,1,1)
    #     levels = np.linspace(-20,20,7)
    #     plt.contourf(*np.meshgrid(KS.t,KS.x), KS.u,levels=levels,cmap=cm.cividis)
    #     ax.set_aspect('equal')
    #     plt.title('u(x,t) contours, init '+label)
    #     plt.xlabel('t')
    #     plt.ylabel('x')

    #     plt.figure('u final')
    #     plt.plot(KS.x, KS.u[:,-1],label=label)
    #     plt.xlabel('x')
    #     plt.ylabel('u')
    #     plt.legend()

if __name__ == '__main__':    
    L = 2*np.pi
    N = 2**8
    dt = 1/220
    T = 15
    nu = 0.1

    KS_FS_sin = different_initial_data(nu=nu, max_wavenumber=N, tmax=T, dt=dt)
    print(KS_FS_sin.u_fourier.shape)


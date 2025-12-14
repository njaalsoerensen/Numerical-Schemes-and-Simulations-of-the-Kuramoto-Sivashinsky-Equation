import os
try:
    os.chdir("Documents/Current_semester/Prosjektoppgave")
except FileNotFoundError: pass

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import KS_base
import gc


class KS_FD(KS_base.KS_equation):
    # Finite difference solver            
    def first_order_forward(self):
        # Solves one step using finite difference
        # Shift to the left 
        u = self.u[self.i_index]
        right_2shift_u = np.roll(u, -2) # Means shifting x_i + 2*k
        right_shift_u = np.roll(u, -1)
        left_shift_u = np.roll(u, 1)
        left_2shift_u = np.roll(u, 2) 
        # Solving for u(x, t+h)
        tot = (right_shift_u - 2*u + left_shift_u)/(self.k**2)
        tot += self.nu*(right_2shift_u - 4*right_shift_u + 6*u - 4*left_shift_u + left_2shift_u)/(self.k**4)
        tot += u*(right_shift_u - left_shift_u)/(2*self.k)
        tot *= -self.h
        
        tot += u
        # Appending solution
        super().append_solve(tot)
    
    def central_time(self):
        if self.i_index < 1:
            self.solve_first_order_forward()
            return
        # Shift to the left 
        u = self.u[self.i_index]
        right_2shift_u = np.roll(u, -2) # Means shifting x_i + 2*k
        right_shift_u = np.roll(u, -1)
        left_shift_u = np.roll(u, 1)
        left_2shift_u = np.roll(u, 2) 
        # Solving for u(x, t+h)
        tot = (right_shift_u - 2*u + left_shift_u)/(self.k**2)
        tot += self.nu*(right_2shift_u - 4*right_shift_u + 6*u - 4*left_shift_u + left_2shift_u)/(self.k**4)
        tot += u*(right_shift_u - left_shift_u)/(2*self.k)
        tot *= -2*self.h
        
        tot += self.u[self.i_index - 1]
        # Appending solution
        super().append_solve(tot)
        
    def second_order_forward(self):
        # Solves one step using finite difference
        if self.i_index < 1:
            self.solve_first_order_forward()
            return
        self.i_index -= 1 # Index shift done for computing with same index as in scheeme
        # Shift to the left 
        u = self.u[self.i_index]
        right_2shift_u = np.roll(u, -2) # Means shifting x_i + 2*k
        right_shift_u = np.roll(u, -1)
        left_shift_u = np.roll(u, 1)
        left_2shift_u = np.roll(u, 2) 
        # Solving for u(x, t+h)
        tot = (right_shift_u - 2*u + left_shift_u)/(self.k**2)
        tot += self.nu*(right_2shift_u - 4*right_shift_u + 6*u - 4*left_shift_u + left_2shift_u)/(self.k**4)
        tot += u*(right_shift_u - left_shift_u)/(2*self.k)
        tot *= 2*self.h
        
        tot -= 3*u
        
        tot += 4*self.u[self.i_index+1]
        # Appending solution
        self.i_index += 1
        super().append_solve(tot)
    

class KS_FT(KS_base.KS_equation):
        
    def __init__(self, nu, L, N, dt, T, u_0, plot_initial=False):
        super().__init__(nu, L, N, dt, T, u_0, plot_initial)
        self.u_hat = np.zeros((self.u.shape[0], int(len(self.u[0]))), dtype=complex)        
        self.u_hat[0] = sp.fft.fft(self.u[0])
        self.u_hat[0][0] = 0 # Should be in initial curve, but still gives slightly off value either way
        
        self.energy_log = [self.compute_energy()]        
        self.energy_conservation_log = [self.compute_energy_conservation()]
        
    def compute_energy(self):
        return np.dot(self.u_hat[self.i_index], np.conjugate(self.u_hat[self.i_index]))/2
    
    def compute_energy_conservation(self):
        tot = 0
        for k in range(1, self.u_hat.shape[1]):
            tot += (k**2 - self.nu*k**4)*(self.u_hat[self.i_index][k]*np.conjugate(self.u_hat[self.i_index][k]))
        return tot
        
    def FT_euler(self):
        tot = np.zeros_like(self.u_hat[0], dtype='complex')
        fft_u2 = sp.fft.fft(self.u[self.i_index]**2)
        for k in range(self.u_hat.shape[1]):
            u_hat_k_next = self.h*(k**2 - self.nu*k**4)*self.u_hat[self.i_index][k]
            u_hat_k_next += self.u_hat[self.i_index][k] # This value ranges from e-11 to e-22
            u_hat_k_next += -self.h*1j*k/2*fft_u2[k] # This value is on the order e-23
            tot[k] = u_hat_k_next
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.ifft(tot).real)
        # plt.plot(self.x_values, tot)
        # plt.show()
        # Expected solution is steady state due to numerical stability issues
        # <href>https://pubs.sciepub.com/ajna/2/3/5/index.html</href>
        
    def ETD1(self):
        # Should also make a method to save the modes on the same way, idk, these could easly be computed though
        # plt.plot(np.arange(len(self.x_values)), self.u_hat[self.i_index])
        # plt.show()        
        tot = np.zeros_like(self.u_hat[0], complex)
        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            exp_qh = np.exp(q*self.h)
            u_hat_k_next = -1j*k/(2*q)*(exp_qh - 1)*fft_u2[k] 
            u_hat_k_next += self.u_hat[self.i_index][k]*exp_qh
            tot[k] = u_hat_k_next
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.irfft(tot)) # 2 times because that seems to give correct size
        # plt.plot(np.arange(len(self.x_values)/2+1), tot)
        # plt.show()
        
    def ETD2(self):
        # Should also make a method to save the modes on the same way, idk, these could easly be computed though
        # plt.plot(np.arange(int(len(self.x_values)/2+1)), self.u_hat[self.i_index])
        # plt.show()
        if self.i_index == 0:
            self.ETD1()
            return

        tot = np.zeros_like(self.u_hat[0], complex)
        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        fft_u2_neg1 = sp.fft.rfft(self.u[self.i_index-1]**2)
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            exp_qh = np.exp(q*self.h)
            # factor = 1j*k/(2)
            
            u_hat_k_next = 0
            
            u_hat_k_next_b = exp_qh - (q*self.h + 1)
            u_hat_k_next_b *= fft_u2[k] - fft_u2_neg1[k]
            u_hat_k_next_b *= -1j*k /(2*self.h*q**2)
            
            u_hat_k_next += -1j*k/(2*q)*(exp_qh - 1)*fft_u2[k]
            
            u_hat_k_next += self.u_hat[self.i_index][k]*exp_qh


            tot[k] = u_hat_k_next + u_hat_k_next_b
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.irfft(tot))
        
    def ETD_trapezoid(self):
        # do EDT1 to get approximating for next timestep then use trapezoid to estimate the nonlinear part
        tot_pos1 = np.zeros_like(self.u_hat[0], complex)
        fft_u2 = sp.fft.fft(self.u[self.i_index]**2)
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            exp_qh = np.exp(q*self.h)
            u_hat_k_next = -1j*k/(2*q)*(exp_qh - 1)*fft_u2[k] 
            u_hat_k_next += self.u_hat[self.i_index][k]*exp_qh
            tot_pos1[k] = u_hat_k_next
        fft_u2_pos1 = sp.fft.ifft(2*tot_pos1.real)
        fft_u2_pos1 = sp.fft.fft(tot_pos1**2)
        tot = np.zeros_like(self.u_hat[0], complex)
        for k in range(1, self.u_hat.shape[1]):
            q = k**2 - self.nu*k**4
            exp_qh = np.exp(q*self.h)
            # u_hat_k_next = -1j*k*self.h/(4*q) * (exp_qh*fft_u2[k] + fft_u2_pos1[k])
            
            u_hat_k_next = -1j*k*self.h/(4) * (exp_qh*fft_u2[k] + fft_u2_pos1[k])
            
            u_hat_k_next += self.u_hat[self.i_index][k]*exp_qh
            tot[k] = u_hat_k_next
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.ifft(tot).real)
        
    def ETD_RK2(self):
        # plt.plot(np.arange(int(len(self.x_values)/2+1)), self.u_hat[self.i_index])
        # plt.show()
        tot = np.zeros_like(self.u_hat[0], complex)
        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        
        phi_1 = lambda z: (np.exp(z)-1)/z
        phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
        
        a_hat = np.zeros_like(self.u_hat[0], complex)
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            a_hat[k] = self.u_hat[self.i_index][k] + self.h*phi_1(q*self.h)*(q*self.u_hat[self.i_index][k] + (-1j*k/2)*fft_u2[k])
        a = sp.fft.irfft(a_hat)
        fft_a2 = sp.fft.rfft(a**2)
        
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            
            u_hat_k_next = a_hat[k] - 1j*k/2*self.h*phi_2(self.h*q)*(-fft_u2[k] + fft_a2[k])
            tot[k] = u_hat_k_next
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.irfft(tot))
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())
        
    def ETD_RK3(self):
        tot = np.zeros_like(self.u_hat[0], complex)
        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        
        phi_1 = lambda z: (np.exp(z)-1)/z
        phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
        phi_3 = lambda z: (np.exp(z) - 1 - z - z**2/2)/(z**3)
        
        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            a_hat[k] = self.u_hat[self.i_index][k] + self.h/2*phi_1(q*self.h/2)*(q*self.u_hat[self.i_index][k] + (-1j*k/2)*fft_u2[k])
        a = sp.fft.irfft(a_hat)
        fft_a2 = sp.fft.rfft(a**2)
        
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            b_hat[k] = self.u_hat[self.i_index][k] + self.h*phi_1(self.h*q)*(q*self.u_hat[self.i_index][k] + (-1j*k/2)*(-fft_u2[k] + 2*fft_a2[k]))
        b = sp.fft.irfft(b_hat)
        fft_b2 = sp.fft.rfft(b**2)
        
        for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
            q = k**2 - self.nu*k**4
            u_hat_k_next = self.u_hat[self.i_index][k]
            u_hat_k_next += self.h*phi_1(self.h*q)*(q*self.u_hat[self.i_index][k] + (-1j*k/2)*fft_u2[k])
            u_hat_k_next += self.h*phi_2(self.h*q)*(-1j*k/2)*(-3*fft_u2[k] + 4*fft_a2[k] - fft_b2[k])
            u_hat_k_next += self.h*phi_3(self.h*q)*(-1j*k/2)*(4*fft_u2[k] - 8*fft_a2[k] + 4*fft_b2[k])
            tot[k] = u_hat_k_next
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.irfft(tot))
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())
        
    # def ETD_RK4(self): # Non-vectorized, have confirmed that solutions concide, vectorized version is 3.5x faster
    #     tot = np.zeros_like(self.u_hat[0], complex)
    #     fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        
    #     phi_1 = lambda z: (np.exp(z)-1)/z
    #     phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
    #     phi_3 = lambda z: (np.exp(z) - 1 - z - z**2/2)/(z**3)
        
    #     a_hat = np.zeros_like(self.u_hat[0], complex)
    #     b_hat = np.zeros_like(self.u_hat[0], complex)
    #     c_hat = np.zeros_like(self.u_hat[0], complex)
    
    #     for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
    #         q = k**2 - self.nu*k**4
    #         a_hat[k] = self.u_hat[self.i_index][k] + self.h/2*phi_1(q*self.h/2)*(q*self.u_hat[self.i_index][k] + (-1j*k/2)*fft_u2[k])
    #     a = sp.fft.irfft(a_hat)
    #     fft_a2 = sp.fft.rfft(a**2)
    
    #     for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
    #         q = k**2 - self.nu*k**4
    #         b_hat[k] = self.u_hat[self.i_index][k] + self.h/2*phi_1(self.h/2*q)*(q*self.u_hat[self.i_index][k] - (-1j*k/2)*fft_a2[k])
    #     b = sp.fft.irfft(b_hat)
    #     fft_b2 = sp.fft.rfft(b**2)
    
    #     for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
    #         q = k**2 - self.nu*k**4
    #         c_hat[k] = a_hat[k] + self.h/2*phi_1(self.h/2*q)*(q*a_hat[k] - (-1j*k/2)*fft_u2[k] + 2*(-1j*k/2)*fft_b2[k])
    #     c = sp.fft.irfft(c_hat)
    #     fft_c2 = sp.fft.rfft(c**2)
    
    #     for k in range(1, self.u_hat.shape[1]): # Were assuming constant frequency is zero without loss of information
    #         q = k**2 - self.nu*k**4
    #         u_hat_k_next = self.u_hat[self.i_index][k]
    #         u_hat_k_next += self.h*phi_1(self.h*q)*(q*self.u_hat[self.i_index][k] + (-1j*k/2)*fft_u2[k])
    #         u_hat_k_next += self.h*phi_2(self.h*q)*(-1j*k/2)*(-3*fft_u2[k] + 2*fft_a2[k] + 2*fft_b2[k] - fft_c2[k])
    #         u_hat_k_next += self.h*phi_3(self.h*q)*(-1j*k/2)*(4*fft_u2[k] - 4*fft_a2[k] - 4*fft_b2[k] + 4*fft_c2[k])
    #         tot[k] = u_hat_k_next
    #     self.u_hat[self.i_index+1] = tot
    #     super().append_solve(sp.fft.irfft(tot))
    #     self.energy_log.append(self.compute_energy())
    #     self.energy_conservation_log.append(self.compute_energy_conservation())
        

    def ETD_RK4(self): # Vectorized version
        fft_u2 = sp.fft.fft(self.u[self.i_index]**2)
        
        phi_1 = lambda z: (np.exp(z)-1)/z
        phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
        phi_3 = lambda z: (np.exp(z) - 1 - z - z**2/2)/(z**3)
        
        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)
        c_hat = np.zeros_like(self.u_hat[0], complex)
        
        k = np.arange( self.u_hat.shape[1])
        q = np.array([k**2 - self.nu*k**4 for k in k])
        print(q)
        print(k)
        print(a_hat.shape)

        a_hat[1:] = self.u_hat[self.i_index][1:] + self.h/2*phi_1(q[1:]*self.h/2)*(q[1:]*self.u_hat[self.i_index][1:] + (-1j*k[1:]/2)*fft_u2[1:])
        a = sp.fft.irfft(a_hat)
        fft_a2 = sp.fft.fft(a)
        print(a.shape)
        print(fft_a2.shape)
        print(a_hat)
        
        b_hat[1:] = self.u_hat[self.i_index][1:] + self.h/2*phi_1(q[1:]*self.h/2)*(q[1:]*self.u_hat[self.i_index][1:] + (-1j*k[1:]/2)*fft_a2[1:])
        b = sp.fft.irfft(b_hat)
        fft_b2 = sp.fft.fft(b**2)  
        
        c_hat[1:] = a_hat[1:] + self.h/2*phi_1(q[1:]*self.h/2)*(q[1:]*a_hat[1:] - (-1j*k[1:]/2)*fft_u2[1:] + 2*(-1j*k[1:]/2)*fft_b2[1:])
        c = sp.fft.irfft(c_hat)
        fft_c2 = sp.fft.fft(c**2)
        
        u_hat_next = self.u_hat[self.i_index]
        # print(self.u_hat[self.i_index][0])
        # print(u_hat_next[0])
        # print()
        u_hat_next[1:] += self.h*phi_1(self.h*q[1:])*(q[1:]*self.u_hat[self.i_index][1:] + (-1j*k[1:]/2)*fft_u2[1:])
        u_hat_next[1:] += self.h*phi_2(self.h*q[1:])*(-1j*k[1:]/2)*(-3*fft_u2[1:] + 2*fft_a2[1:] + 2*fft_b2[1:] - fft_c2[1:])
        u_hat_next[1:] += self.h*phi_3(self.h*q[1:])*(-1j*k[1:]/2)*(4*fft_u2[1:] - 4*fft_a2[1:] - 4*fft_b2[1:] + 4*fft_c2[1:])
        self.u_hat[self.i_index+1] = u_hat_next
        super().append_solve(sp.fft.irfft(self.u_hat[self.i_index+1]))
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())
    


        
        



if __name__ == '__main__':
    gc.collect()        
    
    # L = 2*np.pi
    # N = 2**8
    # dt = 1/400
    # T = 5
    # eq_str = 'np.cos(2*np.pi*x/L) + np.sin(2*np.pi*x/L)' # eq string to be able to save this info
    # u_0 = eval(f"lambda x: {eq_str}")
    
    # nu = 0.3
    
    # Eq_ETD_RK4_Vec = KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=False)  
    # Eq_ETD_RK4_Vec.solve(method='ETD_RK4')
    
    # # <Energy conservation conformation>
    # nu = 0.074
    # nu = 0.114
    
    # Time period attractors comparison with The route to chaos for KS equation
    nu = 0.057 # Good
    nu = 0.0559 # Good
    nu = 0.0555 # Good
    # Steady fully modal attractors
    nu = 0.0397 # Energy is exact same shape, interesting dE vs E, but no comparison
    nu = 0.038 # breaks down after T = 8
    nu = 0.0376
    
    
    eq_str = '-np.sin(x)' # eq string to be able to save this info
    u_0 = eval(f"lambda x: {eq_str}")
    
    # nu = 0.03
    
    L = 2*np.pi
    N = 2**6
    freq = 550
    dt = 1/freq
    T = 15 
    
    Bank = KS_base.DE_bank()
    Eq_ETD_RK4 = KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=False)  
    Eq_ETD_RK4.solve(method='ETD_RK4')
    plt.title(r'Energy vs time for ETD_RK4 scheme, $\nu$ = '+f'{nu}')
    plt.plot(np.linspace(0, T, len(Eq_ETD_RK4.energy_log)), Eq_ETD_RK4.energy_log)
    plt.xlabel('Time (T)')
    plt.ylabel('Energy (E)')
    plt.show()
    
    # np.diff calculates differential as D{y_i} = y_{i+1} - y_i, so need to divide by dt to get dy/dt    
    plt.title(r'Energy conservation equation for ETD_RK4 scheme, $\nu$ = '+f'{nu}')
    plt.plot(np.linspace(0, T, len(Eq_ETD_RK4.energy_log)-1), np.diff(Eq_ETD_RK4.energy_log)/(dt), label='Energy differential LHS') # LHS
    plt.plot(np.linspace(0, T, len(Eq_ETD_RK4.energy_conservation_log)), Eq_ETD_RK4.energy_conservation_log, linestyle='--', label='Energy differential RHS') # RHS
    plt.xlabel('Time (T)')
    plt.ylabel('Energy differential (dE)')
    plt.legend()
    plt.show()
    
    in_time = 5
    plt.title(r'Energy vs Energy differentialETD_RK4 scheme, $\nu$ = '+f'{nu}')
    
    # print(np.array(Eq_ETD_RK4.energy_log).shape)
    # print(np.diff(Eq_ETD_RK4.energy_log).shape)
    
    # plt.scatter(np.array(Eq_ETD_RK4.energy_log)[5000+1:], np.diff(Eq_ETD_RK4.energy_log)[5000:]/(dt))
    plt.scatter(np.array(Eq_ETD_RK4.energy_log)[(freq*in_time)+1:][0], np.diff(Eq_ETD_RK4.energy_log)[(freq*in_time):][0]/(dt), label='Start')
    plt.plot(np.array(Eq_ETD_RK4.energy_log)[(freq*in_time)+1:], np.diff(Eq_ETD_RK4.energy_log)[(freq*in_time):]/(dt))
    plt.xlabel('Energy (E)')
    plt.ylabel('Energy differential (dE)')
    plt.legend()
    plt.show()
    # # </Energy consrvation confirmation>
    
    # Bank.add_DE('Eq_ETD_RK4', Eq_ETD_RK4)
    
    
        
    # # Finite Difference
    # nu = 0.02
    # L = 2*np.pi
    # N = 2**6
    # # dt = nu/((N/3)**4) # Works for any value nu This value matters alot, it has a very narrow range of values for stability
    # dt = nu/((N/7)**4) # Works for low values of nu < 0.05
    # T = 1
    # u_0 = lambda x:  np.cos((2*np.pi*x)/L)
    # u_1 = lambda x: 1/10*np.cos(2*(4*np.pi*x)/L) + np.sin((2*np.pi*x)/L)
    
    
    # Bank = KS_base.DE_bank()
    # Eq_fd_1 = KS_FD(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=True)  
    # Eq_fd_2 = KS_FD(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_1, plot_initial=True)
    # diff_names = [(Eq_fd_1, 'Eq_fd_1'), (Eq_fd_2, 'Eq_fd_2')]
    # for diff in diff_names:
    #     Bank.add_DE(diff[0], diff[1])
    #     Bank.bank[diff[1]].solve('first_order_forward')
    
    # Bank.plot_solve_history(diff_names[0][1], step_speed=1000, N_simul=20000)
    # Eq_fd_1.reduce_shape(frac=1/5)
    # Eq_fd_2.reduce_shape(frac=1/5)
    # Bank.plot_solve_history(diff_names[0][1], step_speed=1000, N_simul=int(20000/5))
            
    # Bank.plot_solve((diff_names[0][1], diff_names[1][1]), labs=True, step_speed=1000)
            
    
    # # Fourier Transform    
    # # FT is working as itended (described in link), it's expected to give steady state solutions
    # nu = 0.025
    # L = 2*np.pi
    # N = 2**7
    # dt = nu/((N/4)**4)
    # T = 0.0005
    # u_0 = lambda x:  np.cos(3*(2*np.pi*x)/L) + np.sin(4*(2*np.pi*x)/L)
    
    # Eq_ft_1 = KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=True)   
    # Bank = KS_base.DE_bank()
    # Bank.add_DE(Eq_ft_1, 'Eq_ft_1')
    # Eq_ft_1.solve(method='FT_euler')
    
    # Bank.plot_solve(['Eq_ft_1'], labs=True, title=0, step_speed=1)
    
  
    # Bank.plot_solve(['Eq_ETD_RK3', 'Eq_ETD_RK4'], labs=True, title=0, step_speed=10)
    
    # Bank.plot_solve(['Eq_ETD_RK3', 'Eq_ETD_RK4'], labs=True, title=0, step_speed=4, mode='fourier')
    
    # Bank.plot_solve(['Eq_ETD_RK4'], labs=True, title=0, step_speed=4, mode='fourier')
    
    # Bank.plot_solve_history('Eq_EDT1_1', step_speed=40, N_simul=3000)

    
    # # # Finite Difference
    # nu = 0.02
    # L = 2*np.pi
    # N = 2**6
    # # dt = nu/((N/3)**4) # Works for any value nu This value matters alot, it has a very narrow range of values for stability
    # dt = nu/((N/7)**4) # Works for low values of nu < 0.05
    # T = 5000*dt
    # u_0 = lambda x:  np.cos((2*np.pi*x)/L)
    # # u_1 = lambda x: 1/10*np.cos(2*(4*np.pi*x)/L) + np.sin((2*np.pi*x)/L)
        
    # Eq_fd_1 = KS_FD(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=True)
    # Bank.add_DE(Eq_fd_1, 'Eq_fd_1')
    # Eq_fd_1.solve('first_order_forward')
    
    # Bank.plot_solve(['Eq_EDT1_1', 'Eq_fd_1'], labs=True, title=0, step_speed=2)
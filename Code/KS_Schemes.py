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
        self.u_hat = np.zeros((self.u.shape[0], int(len(self.u[0])/2)+1), dtype=complex)        
        self.u_hat[0] = sp.fft.rfft(self.u[0])
        self.u_hat[0][0] = 0 # Should be in initial curve, but still gives slightly off value either way
        self.bruh = []
        
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
        self.u_hat[self.i_index+1] = self.handle_tol(tot)
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
        self.u_hat[self.i_index+1] = self.handle_tol(tot)
        super().append_solve(sp.fft.irfft(tot))
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())
        

    def ETD_RK4(self): # Vectorized version

        phi_1 = lambda z: (np.exp(z)-1)/z
        phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
        phi_3 = lambda z: (np.exp(z) - 1 - z - (z**2)/2)/(z**3)
        
        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)
        c_hat = np.zeros_like(self.u_hat[0], complex)
        u_hat_next = np.zeros_like(self.u_hat[0], complex)
        
        k = np.arange(self.u_hat.shape[1])
        q = np.array([k**2 - self.nu*k**4 for k in k]) 
        # Cox and Matthews p3 argue KS equation is dissapative
        # https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999100X01325/1-s2.0-S0021999102969950/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQD9nf%2BbpI0g7XzvYxpZrO9jbVDz05YgkcXU0kncB1sWkQIhAOzPG42k%2FwlW6mBi07P8QrsMRVjxxD3oQOWpaL5Z%2BxaNKrsFCKn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgycmOWPT7VXUZXTEJwqjwWjBlUiiHqGhTGkw6y2TLXq%2FI1kOwjaSMavao0q89e4spk9XK1hw21akZrp5tY%2BwXvjPHNVC9uWBV%2BwQ%2FEviA6Iee9%2BFg0a4WkBnWtul3J2fGCnvokOhbRaz4rGVLyDezExEN20%2FZFtOP5kOGjz6sA9n7Gt8ogvwb2BV%2F8qONzH9WQLLJ%2BBm9CAlzPFUrfco28%2FrDbUsYHPqM5cE0%2BGo0yQcaQJ3baLBYzx6kodp%2FNgrL%2FowAZqfx84hRPllZJ7g8e3JhXpyMrER50Inv3g%2BdBWeyagn%2FSZ70xefgDG2sCAPnHl9q6BsXHNi9dCQj3J3ZFkv7WR47KhoWflibYhxGl%2FJ1sLLardZtI%2BH4xeSLsaJCZN7Blqq79Nk0K2HM6HYiCW8kV6lOz4pL5XGf%2F%2FiJiMVxGBYaixLJiymB5YoUzysyzvWVIzXgaYFEKKgnw7EB4D8H%2BgIn0ccxO%2BEOI39V751H5cytNiEM6LwKCme57HgLWJaTujGO3wweBXBVANNlLmPH%2BaVVNHnjehKZmw3fArxs6XDXsYfpIxDECjAZIP%2FzFLiTekzwaW417M4o0R53AlNMWKkvp5NrAXcNKrDgbWl1rs8G%2F75ja4fV0CcHrSNx81w1pmun%2FzGvEpBEJ9j8In3kX3aPAWY%2FdcI%2FtURLQyE%2B3E59zLQWBAXjsoX%2FXMIFGsNC4PRLHoHCdiB8BPSQntbshmwtIbUOCGJjM4KfbkkKwVGhwt%2FzsEQSaReueWPqrkkwg2G6XxT5JbDAiu1qmy3gUBuYDFv9xN7r8yStVx2RVIj77kjeBSmXiW5Ps8g6otBKKjU15NYKb7%2F%2BcJiQFkOu5wcitccJlcI%2BTzrIlAK97mcbmScgpbFNJyAk83MNzEo8kGOrABAboP7T%2Fu0FdReJcIm%2Bl%2B3CE2PmQEDMsgrpRTz%2ByjBjgdH04aodgZwuxwf2eRKx4rGjqO8igpMUfzbrBEoonWOoSt5TH9i8b2Bz4n2G7QJWoJEDWYyuz8psSOWc3O5Y89zn4CPV4IUap2ZjG%2BLEdtGe4cff9lbCMaVL2PIzHVZYg2vUCG%2BSh%2F6Pe%2FXoGCStYP8b4KtmCGpKTfAVXOzLAMKDBjM4CHOp95atizOsau8Z0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251128T004042Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYROBPL2KS%2F20251128%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=51d7d5f98583d3d78339f8736d4915d5057738f4a01c4d62e7e323fdf62578bf&hash=dda47cf22bf42b5f73d0b5c61ead2ca148296f0934461a71c99e5a5566c779cb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0021999102969950&tid=spdf-98c936dc-2f36-4f22-9ca6-75d2c36321a8&sid=9352dcf86b195544126a29e010fa62ec6343gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080c5d0f505400500454&rr=9a55d0c01a2a56a4&cc=no
        
        C_non = -1j*k[1:]/2 # Nonlinear constant # Weird how this can be negative or postive with no change, linear is in charge of energy transfere

        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
    
        a_hat[1:] = self.u_hat[self.i_index][1:] + self.h/2*phi_1(q[1:]*self.h/2)*(q[1:]*self.u_hat[self.i_index][1:] + C_non*fft_u2[1:])
        a = sp.fft.irfft(a_hat)
        fft_a2 = sp.fft.rfft(a**2)

        
        b_hat[1:] = self.u_hat[self.i_index][1:] + self.h/2*phi_1(q[1:]*self.h/2)*(q[1:]*self.u_hat[self.i_index][1:] + C_non*fft_a2[1:])
        b = sp.fft.irfft(b_hat)
        fft_b2 = sp.fft.rfft(b**2) 
        
        c_hat[1:] = a_hat[1:] + self.h/2*phi_1(q[1:]*self.h/2)*(q[1:]*a_hat[1:] - C_non*fft_u2[1:] + 2*C_non*fft_b2[1:])
        c = sp.fft.irfft(c_hat)
        fft_c2 = sp.fft.rfft(c**2)
        
        u_hat_next += self.u_hat[self.i_index]

        u_hat_next[1:] += self.h*phi_1(self.h*q[1:])*(q[1:]*self.u_hat[self.i_index][1:] + C_non*fft_u2[1:])
        u_hat_next[1:] += self.h*phi_2(self.h*q[1:])*C_non*(-3*fft_u2[1:] + 2*fft_a2[1:] + 2*fft_b2[1:] - fft_c2[1:])
        u_hat_next[1:] += self.h*phi_3(self.h*q[1:])*C_non*(4*fft_u2[1:] - 4*fft_a2[1:] - 4*fft_b2[1:] + 4*fft_c2[1:])        
        
        if self.i_index % 10 == 0:
            mode_energy = (u_hat_next*np.conjugate(u_hat_next)).real
            self.bruh.append(mode_energy[2])
            # N = 20
            # plt.plot(np.arange(len(mode_energy))[0:N], mode_energy[0:N])
            # plt.scatter(np.arange(len(mode_energy))[0:N], mode_energy[0:N])            
            # plt.xlabel('Mode')
            # plt.ylabel('Energy')
            # plt.show()
        self.u_hat[self.i_index+1] = self.handle_tol(u_hat_next)
        super().append_solve(sp.fft.irfft(self.u_hat[self.i_index+1]))
        
        # if self.i_index % 5 == 0:
        #     plt.plot(np.linspace(0, 2*np.pi, len(self.u[self.i_index])), self.u[self.i_index])
        #     plt.xlabel('x_values')
        #     plt.ylabel('y_values')
        #     plt.show()
            
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())
        
        
    def ETD_RK4_CM(self): # Vectorized version

        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)
        c_hat = np.zeros_like(self.u_hat[0], complex)
        u_hat_next = np.zeros_like(self.u_hat[0], complex)
        k = np.arange(self.u_hat.shape[1])[1:] # Don't need first index for any computation
        q = np.array([k**2 - self.nu*k**4 for k in k]) 
        # Cox and Matthews p3 argue KS equation is dissapative
        # https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999100X01325/1-s2.0-S0021999102969950/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQD9nf%2BbpI0g7XzvYxpZrO9jbVDz05YgkcXU0kncB1sWkQIhAOzPG42k%2FwlW6mBi07P8QrsMRVjxxD3oQOWpaL5Z%2BxaNKrsFCKn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgycmOWPT7VXUZXTEJwqjwWjBlUiiHqGhTGkw6y2TLXq%2FI1kOwjaSMavao0q89e4spk9XK1hw21akZrp5tY%2BwXvjPHNVC9uWBV%2BwQ%2FEviA6Iee9%2BFg0a4WkBnWtul3J2fGCnvokOhbRaz4rGVLyDezExEN20%2FZFtOP5kOGjz6sA9n7Gt8ogvwb2BV%2F8qONzH9WQLLJ%2BBm9CAlzPFUrfco28%2FrDbUsYHPqM5cE0%2BGo0yQcaQJ3baLBYzx6kodp%2FNgrL%2FowAZqfx84hRPllZJ7g8e3JhXpyMrER50Inv3g%2BdBWeyagn%2FSZ70xefgDG2sCAPnHl9q6BsXHNi9dCQj3J3ZFkv7WR47KhoWflibYhxGl%2FJ1sLLardZtI%2BH4xeSLsaJCZN7Blqq79Nk0K2HM6HYiCW8kV6lOz4pL5XGf%2F%2FiJiMVxGBYaixLJiymB5YoUzysyzvWVIzXgaYFEKKgnw7EB4D8H%2BgIn0ccxO%2BEOI39V751H5cytNiEM6LwKCme57HgLWJaTujGO3wweBXBVANNlLmPH%2BaVVNHnjehKZmw3fArxs6XDXsYfpIxDECjAZIP%2FzFLiTekzwaW417M4o0R53AlNMWKkvp5NrAXcNKrDgbWl1rs8G%2F75ja4fV0CcHrSNx81w1pmun%2FzGvEpBEJ9j8In3kX3aPAWY%2FdcI%2FtURLQyE%2B3E59zLQWBAXjsoX%2FXMIFGsNC4PRLHoHCdiB8BPSQntbshmwtIbUOCGJjM4KfbkkKwVGhwt%2FzsEQSaReueWPqrkkwg2G6XxT5JbDAiu1qmy3gUBuYDFv9xN7r8yStVx2RVIj77kjeBSmXiW5Ps8g6otBKKjU15NYKb7%2F%2BcJiQFkOu5wcitccJlcI%2BTzrIlAK97mcbmScgpbFNJyAk83MNzEo8kGOrABAboP7T%2Fu0FdReJcIm%2Bl%2B3CE2PmQEDMsgrpRTz%2ByjBjgdH04aodgZwuxwf2eRKx4rGjqO8igpMUfzbrBEoonWOoSt5TH9i8b2Bz4n2G7QJWoJEDWYyuz8psSOWc3O5Y89zn4CPV4IUap2ZjG%2BLEdtGe4cff9lbCMaVL2PIzHVZYg2vUCG%2BSh%2F6Pe%2FXoGCStYP8b4KtmCGpKTfAVXOzLAMKDBjM4CHOp95atizOsau8Z0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251128T004042Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYROBPL2KS%2F20251128%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=51d7d5f98583d3d78339f8736d4915d5057738f4a01c4d62e7e323fdf62578bf&hash=dda47cf22bf42b5f73d0b5c61ead2ca148296f0934461a71c99e5a5566c779cb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0021999102969950&tid=spdf-98c936dc-2f36-4f22-9ca6-75d2c36321a8&sid=9352dcf86b195544126a29e010fa62ec6343gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080c5d0f505400500454&rr=9a55d0c01a2a56a4&cc=no
        
        C_non = -1j*k/2 # Nonlinear constant # Weird how this can be negative or postive with no change, linear is in charge of energy transfere
        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        N_u = C_non*fft_u2[1:] # Nonlinear opperator on u for values used in computation
        
        phi = lambda z: (np.exp(z) - 1)/q
        # phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
        # phi_3 = lambda z: (np.exp(z) - 1 - z - (z**2)/2)/(z**3)
        # C is q and h is h
        
        phi2 = phi(q*self.h/2)
        phi = phi(q*self.h)
        
        xp2 = np.exp(q*self.h/2)
        xp = np.exp(q*self.h)
        
        a_hat[1:] = self.u_hat[self.i_index][1:]*xp2 + phi2*N_u
        a = sp.fft.irfft(a_hat)
        fft_a2 = sp.fft.rfft(a**2)
        N_a = C_non*fft_a2[1:]
        
        b_hat[1:] = self.u_hat[self.i_index][1:]*xp2 + phi2*N_a
        b = sp.fft.irfft(b_hat)
        fft_b2 = sp.fft.rfft(b**2) 
        N_b = C_non*fft_b2[1:]
        
        c_hat[1:] = a_hat[1:]*xp2 + phi2*(2*N_b - N_u)
        c = sp.fft.irfft(c_hat)
        fft_c2 = sp.fft.rfft(c**2)
        N_c = C_non*fft_c2[1:]

        u_hat_next[1:] += N_u*(-4 - q*self.h + xp*(4 - 3*q*self.h + self.h**2*q**2))
        u_hat_next[1:] += 2*(N_a + N_b)*(2 + q*self.h + xp*(-2 + q*self.h))
        u_hat_next[1:] += N_c*(-4 - 3*q*self.h - q**2*self.h**2 + xp*(4 - q*self.h))
        u_hat_next[1:] /= q**3*self.h**2
        u_hat_next[1:] += self.u_hat[self.i_index][1:]*xp
        
        # print((u_hat_next.imag))
        # Max false real value was 1.5 e-14
        # Imag values for modes with energy are consentrated from 1e+2 to 1.6e-4 with next value 2.6e-7, 4.6e-10, 6.9e-13, then machine epsilon for rest
        # tol = 1.5e-14
        # u_hat_next.imag = np.where(abs(u_hat_next.imag) < tol, 0, u_hat_next.imag)
        # u_hat_next.real = np.where(abs(u_hat_next.real) < tol, 0, u_hat_next.real)
        
        if self.i_index % 10 == 0:
            mode_energy = (u_hat_next*np.conjugate(u_hat_next)).real
            self.bruh.append(mode_energy[2])
            # N = 20
            # plt.plot(np.arange(len(mode_energy))[0:N], mode_energy[0:N])
            # plt.scatter(np.arange(len(mode_energy))[0:N], mode_energy[0:N])            
            # plt.xlabel('Mode')
            # plt.ylabel('Energy')
            # plt.show()
        self.u_hat[self.i_index+1] = self.handle_tol(u_hat_next)
        super().append_solve(sp.fft.irfft(self.u_hat[self.i_index+1]))
        
        # if self.i_index % 5 == 0:
        #     plt.plot(np.linspace(0, 2*np.pi, len(self.u[self.i_index])), self.u[self.i_index])
        #     plt.xlabel('x_values')
        #     plt.ylabel('y_values')
        #     plt.show()
            
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())
    
        # Does not work at all T_T
        
        
    def handle_tol(self, arr, tol=1.5e-13): # Important for dealing with rounding errors
        arr.imag = np.where(abs(arr.imag) < tol, 0, arr.imag)
        arr.real = np.where(abs(arr.real) < tol, 0, arr.real)
        return arr
    
        

        
        



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
    
    # # <Plot solve history>

    
    # eq_str = '-np.sin(x)' # eq string to be able to save this info
    # u_0 = eval(f"lambda x: {eq_str}")
    # nu = 0.0377
    
    # L = 2*np.pi
    # N = 2**8
    # freq = 250
    # dt = 1/freq
    # T = 12
    
    # in_time = 5
    
    # Bank = KS_base.DE_bank()
    # solve_method = 'ETD_RK4'
    # Eq_ETD_RK4 = KS_FT(nu=nu, L=L, N=N, dt=dt, T=T, u_0=u_0, plot_initial=False)  
    # Eq_ETD_RK4.solve(method=solve_method)
    
    # Bank.add_DE('Eq_ETD_RK4', Eq_ETD_RK4)
    # Bank.plot_solve_history('Eq_ETD_RK4', title=f'{solve_method} solution, '+r'$\nu$ = '+f'{nu}', lab=0, step_speed=200, N_simul=1000, ani=False)

    # # </Plot solve history>
    
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
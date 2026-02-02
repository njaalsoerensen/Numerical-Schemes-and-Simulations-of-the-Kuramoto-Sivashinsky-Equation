import os
try:
    os.chdir("Documents/Current_semester/Master")
except FileNotFoundError: pass

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
        # self.bruh = []
        
        self.energy_log = [self.compute_energy()]        
        self.energy_conservation_log = [self.compute_energy_conservation()]
        
        # Memoization of common computational values
        k = np.arange(self.u_hat.shape[1])[1:]
        self.C_non = -1j*k/2
        self.q_compute = np.array([k**2 - self.nu*k**4 for k in k])
        self.q_compute_power_2 = self.q_compute**2
        self.q_compute_power_3 = self.q_compute**3
        self.q_compute_h = self.q_compute*self.h
        self.q_compute_power_2_h_power_2 = self.q_compute_power_2*self.h**2
        self.q_compute_power_3_h_power_2 = self.q_compute_power_3*self.h**2
        
        # print(self.q_compute)
        # if 0 not in self.q_compute:
        phi_1 = lambda z: (np.exp(z)-1)/z
        phi_2 = lambda z: (np.exp(z) - 1 - z)/(z**2)
        phi_3 = lambda z: (np.exp(z) - 1 - z - z**2/2)/(z**3)
        def phi_10(z):
            if z == 0:
                print(z)
                return np.exp(z)/(1)
            return (np.exp(z)-1)/z
        def phi_20(z):
            if z == 0:
                return np.exp(z)/(2*1)
            return (np.exp(z) - 1 - z)/(z**2)
        def phi_30(z):
            if z == 0:
                return np.exp(z)/(3*2*1)
            return (np.exp(z) - 1 - z - z**2/2)/(z**3)
        S_1 = self.q_compute_h/2
        self.phi_1_common = self.h/2*np.where(S_1 != 0, phi_1(S_1), 1)
        self.phi_1_common_V2 = self.h*np.where(self.q_compute_h != 0, phi_1(self.q_compute_h), 1)
        self.phi_2_common_V2 = self.h*np.where(self.q_compute_h != 0, phi_2(self.q_compute_h),1/2)
        self.phi_3_common_V2 = self.h*np.where(self.q_compute_h != 0, phi_3(self.q_compute_h), 1/6)
        
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
            u_hat_k_next = -1j*k*self.h/(4) * (exp_qh*fft_u2[k] + fft_u2_pos1[k])
            
            u_hat_k_next += self.u_hat[self.i_index][k]*exp_qh
            tot[k] = u_hat_k_next
        self.u_hat[self.i_index+1] = tot
        super().append_solve(sp.fft.ifft(tot).real)
        
    def ETD_RK2(self):
        a_hat = np.zeros_like(self.u_hat[0], complex)
        u_hat_next = np.zeros_like(self.u_hat[0], complex)
        
        u_hat = self.u_hat[self.i_index][1:]
        N_u = self.C_non*sp.fft.rfft(self.u[self.i_index]**2)[1:]
        q_u = self.q_compute*u_hat

        a_hat[1:] = u_hat + self.phi_1_common_V2*(q_u + N_u)
        a = sp.fft.irfft(a_hat)
        N_a = self.C_non*sp.fft.rfft(a**2)[1:]
        
        u_hat_next[1:] = a_hat[1:] + self.phi_2_common_V2*(-N_u + N_a)
        self.solved(u_hat_next)

    def ETD_RK3(self):
        u_hat_next = np.zeros_like(self.u_hat[0], complex)        
        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)

        u_hat = self.u_hat[self.i_index][1:]
        N_u = self.C_non*sp.fft.rfft(self.u[self.i_index]**2)[1:]
        q_u = self.q_compute*u_hat
        
        a_hat[1:] = u_hat + self.phi_1_common*(q_u + N_u)
        a = sp.fft.irfft(a_hat)
        N_a = self.C_non*sp.fft.rfft(a**2)[1:]
        
        b_hat[1:] = u_hat + self.phi_1_common_V2*(q_u - N_u + 2*N_a)
        b = sp.fft.irfft(b_hat)
        N_b = self.C_non*sp.fft.rfft(b**2)[1:]

        u_hat_next[1:] += u_hat + self.phi_1_common_V2*(q_u + N_u)
        u_hat_next[1:] += self.phi_2_common_V2*(-3*N_u + 4*N_a - N_b)
        u_hat_next[1:] += self.phi_3_common_V2*(4*N_u - 8*N_a + 4*N_b)
        
        self.solved(u_hat_next)
        
    def ETD_RK4(self): # Vectorized version        
        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)
        c_hat = np.zeros_like(self.u_hat[0], complex)
        u_hat_next = np.zeros_like(self.u_hat[0], complex)
        
        u_hat = self.u_hat[self.i_index][1:]
        N_u = self.C_non*sp.fft.rfft(self.u[self.i_index]**2)[1:]
        q_u = self.q_compute*u_hat
        
        a_hat[1:] = u_hat + self.phi_1_common*(q_u + N_u)
        a = sp.fft.irfft(a_hat)
        N_a = self.C_non*sp.fft.rfft(a**2)[1:]
        
        b_hat[1:] = u_hat + self.phi_1_common*(q_u + N_a)
        b = sp.fft.irfft(b_hat)
        N_b = self.C_non*sp.fft.rfft(b**2)[1:]
        
        c_hat[1:] = a_hat[1:] + self.phi_1_common*(self.q_compute*a_hat[1:] - N_u + 2*N_b)
        c = sp.fft.irfft(c_hat)
        N_c =self.C_non*sp.fft.rfft(c**2)[1:]
        
        u_hat_next[1:] += u_hat + self.phi_1_common_V2*(q_u + N_u)
        u_hat_next[1:] += self.phi_2_common_V2*(-3*N_u + 2*N_a + 2*N_b - N_c)
        u_hat_next[1:] += self.phi_3_common_V2*(4*N_u - 4*N_a - 4*N_b + 4*N_c)
        self.solved(u_hat_next)

    def ETD_RK4_CM(self): # Vectorized version

        a_hat = np.zeros_like(self.u_hat[0], complex)
        b_hat = np.zeros_like(self.u_hat[0], complex)
        c_hat = np.zeros_like(self.u_hat[0], complex)
        u_hat_next = np.zeros_like(self.u_hat[0], complex)

        # Cox and Matthews p3 argue KS equation is dissapative
        # https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999100X01325/1-s2.0-S0021999102969950/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQD9nf%2BbpI0g7XzvYxpZrO9jbVDz05YgkcXU0kncB1sWkQIhAOzPG42k%2FwlW6mBi07P8QrsMRVjxxD3oQOWpaL5Z%2BxaNKrsFCKn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgycmOWPT7VXUZXTEJwqjwWjBlUiiHqGhTGkw6y2TLXq%2FI1kOwjaSMavao0q89e4spk9XK1hw21akZrp5tY%2BwXvjPHNVC9uWBV%2BwQ%2FEviA6Iee9%2BFg0a4WkBnWtul3J2fGCnvokOhbRaz4rGVLyDezExEN20%2FZFtOP5kOGjz6sA9n7Gt8ogvwb2BV%2F8qONzH9WQLLJ%2BBm9CAlzPFUrfco28%2FrDbUsYHPqM5cE0%2BGo0yQcaQJ3baLBYzx6kodp%2FNgrL%2FowAZqfx84hRPllZJ7g8e3JhXpyMrER50Inv3g%2BdBWeyagn%2FSZ70xefgDG2sCAPnHl9q6BsXHNi9dCQj3J3ZFkv7WR47KhoWflibYhxGl%2FJ1sLLardZtI%2BH4xeSLsaJCZN7Blqq79Nk0K2HM6HYiCW8kV6lOz4pL5XGf%2F%2FiJiMVxGBYaixLJiymB5YoUzysyzvWVIzXgaYFEKKgnw7EB4D8H%2BgIn0ccxO%2BEOI39V751H5cytNiEM6LwKCme57HgLWJaTujGO3wweBXBVANNlLmPH%2BaVVNHnjehKZmw3fArxs6XDXsYfpIxDECjAZIP%2FzFLiTekzwaW417M4o0R53AlNMWKkvp5NrAXcNKrDgbWl1rs8G%2F75ja4fV0CcHrSNx81w1pmun%2FzGvEpBEJ9j8In3kX3aPAWY%2FdcI%2FtURLQyE%2B3E59zLQWBAXjsoX%2FXMIFGsNC4PRLHoHCdiB8BPSQntbshmwtIbUOCGJjM4KfbkkKwVGhwt%2FzsEQSaReueWPqrkkwg2G6XxT5JbDAiu1qmy3gUBuYDFv9xN7r8yStVx2RVIj77kjeBSmXiW5Ps8g6otBKKjU15NYKb7%2F%2BcJiQFkOu5wcitccJlcI%2BTzrIlAK97mcbmScgpbFNJyAk83MNzEo8kGOrABAboP7T%2Fu0FdReJcIm%2Bl%2B3CE2PmQEDMsgrpRTz%2ByjBjgdH04aodgZwuxwf2eRKx4rGjqO8igpMUfzbrBEoonWOoSt5TH9i8b2Bz4n2G7QJWoJEDWYyuz8psSOWc3O5Y89zn4CPV4IUap2ZjG%2BLEdtGe4cff9lbCMaVL2PIzHVZYg2vUCG%2BSh%2F6Pe%2FXoGCStYP8b4KtmCGpKTfAVXOzLAMKDBjM4CHOp95atizOsau8Z0%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251128T004042Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYROBPL2KS%2F20251128%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=51d7d5f98583d3d78339f8736d4915d5057738f4a01c4d62e7e323fdf62578bf&hash=dda47cf22bf42b5f73d0b5c61ead2ca148296f0934461a71c99e5a5566c779cb&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0021999102969950&tid=spdf-98c936dc-2f36-4f22-9ca6-75d2c36321a8&sid=9352dcf86b195544126a29e010fa62ec6343gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=080c5d0f505400500454&rr=9a55d0c01a2a56a4&cc=no
        
        fft_u2 = sp.fft.rfft(self.u[self.i_index]**2)
        N_u = self.C_non*fft_u2[1:] # Nonlinear opperator on u for values used in computation
        
        phi = lambda z: (np.exp(z) - 1)/self.q_compute
        
        u_hat = self.u_hat[self.i_index][1:] 
        
        phi2 = phi(self.q_compute_h/2)
        phi = phi(self.q_compute_h)
        
        xp2 = np.exp(self.q_compute_h/2)
        xp = np.exp(self.q_compute_h)
        
        a_hat[1:] = u_hat*xp2 + phi2*N_u
        a = sp.fft.irfft(a_hat)
        fft_a2 = sp.fft.rfft(a**2)
        N_a = self.C_non*fft_a2[1:]
        
        b_hat[1:] = u_hat*xp2 + phi2*N_a
        b = sp.fft.irfft(b_hat)
        fft_b2 = sp.fft.rfft(b**2) 
        N_b = self.C_non*fft_b2[1:]
        
        c_hat[1:] = a_hat[1:]*xp2 + phi2*(2*N_b - N_u)
        c = sp.fft.irfft(c_hat)
        fft_c2 = sp.fft.rfft(c**2)
        N_c = self.C_non*fft_c2[1:]

        u_hat_next[1:] += N_u*(-4 - self.q_compute_h + xp*(4 - 3*self.q_compute_h + self.q_compute_power_2_h_power_2))
        u_hat_next[1:] += 2*(N_a + N_b)*(2 + self.q_compute_h + xp*(-2 + self.q_compute_h))
        u_hat_next[1:] += N_c*(-4 - 3*self.q_compute_h - self.q_compute_power_2_h_power_2 + xp*(4 - self.q_compute_h))
        u_hat_next[1:] /= self.q_compute_power_3_h_power_2
        u_hat_next[1:] += u_hat*xp
        self.solved(u_hat_next)

    def solved(self, solve):
        self.u_hat[self.i_index+1] = self.handle_tol(solve)
        super().append_solve(sp.fft.irfft(self.u_hat[self.i_index+1]))
        self.energy_log.append(self.compute_energy())
        self.energy_conservation_log.append(self.compute_energy_conservation())     

    def handle_tol(self, arr, tol=1.5e-13): # Important for dealing with rounding errors
        arr.imag = np.where(abs(arr.imag) < tol, 0, arr.imag)
        arr.real = np.where(abs(arr.real) < tol, 0, arr.real)
        return arr



if __name__ == '__main__':
    gc.collect()        
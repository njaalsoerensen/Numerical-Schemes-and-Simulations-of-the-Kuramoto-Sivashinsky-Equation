import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.signal import convolve
import numpy as np


def run_KS_finite_wavenumber(nu, max_wavenumber, init, tmax, dt):
    """
    Solve the KS equation:
    d_t u(x,t) + d_xx u(x,t) + nu d_xxxx u(x,t) + u(x,t) d_x u(x,t) = 0
    u(x+2pi,t) = u(x,t)
    u(x,0) = init(x)
    using a fourier expansion.

    Input:
        nu: hyperviscosity parameter
        max_wavenumber: Largest wavenumber, corresponds to the nyquist frequency
        init: initial profile.
        tmax: maximal integration time
        dt: time step of output
    Output:
        res: Dictionary of solve_ivp result. In particular,
        res.t contains the time points of the evaluation and
        res.y is a (max_wavenumber, res.t.size) nump array.
    """
    # frequencies
    K = np.arange(1.0, max_wavenumber + 1.0)  # /(2*np.pi)
    g = K**2 - nu * K**4

    def state_tmp(state):
        # make the 2N+1-size vector -fN vN^*, -f(N-1) v(N-1)^*, ..., f(N-1) v(N-1), fN vN
        tmp_w = np.zeros(2 * state.size + 1, dtype="complex")
        tmp_w[: state.size] = -(K * np.conj(state))[::-1]
        tmp_w[state.size + 1 :] = K * state

        tmp_v = np.zeros(2 * state.size + 1, dtype="complex")
        tmp_v[: state.size] = np.conj(state)[::-1]
        tmp_v[state.size + 1 :] = state

        return tmp_w, tmp_v

    # State vector is k=1 to k=max_wavenumber
    def KS_system(time, state, dealias=False):
        # After some testing, difference between aliased and dealiased states are negligible.
        tmp_w, tmp_v = state_tmp(state)

        if dealias:
            tmp_w = np.pad(tmp_w, state.size)
            tmp_v = np.pad(tmp_v, state.size)

            start = 2 * state.size + 1
            stop = -state.size

        else:
            start = state.size + 1
            stop = None

        res = g * state - 1.0j * convolve(tmp_w, tmp_v, mode="same")[start:stop]

        return res

    # Some useful functions to compute the jacobian
    F = sp.coo_matrix(K)
    F = sp.vstack([F for i in range(max_wavenumber)])

    indices = np.arange(max_wavenumber - 1, -max_wavenumber, -1)

    G = sp.spdiags(g, 0, g.size, g.size, format="coo")

    def KS_jac(time, state):
        tmp = state_tmp(state)[1][1:-1]

        W = sp.diags(tmp, indices, format="coo")

        return (G - 1.0j * F.multiply(W)).toarray()

    return solve_ivp(
        KS_system,
        [0, tmax],
        init,
        method="BDF",
        jac=KS_jac,
        t_eval=np.arange(0, tmax, dt),
        atol=1e-8,
        rtol=1e-8,
    )


def run_KS_finite_wavenumber_sine(nu, max_wavenumber, init, tmax, dt):
    """
    Solve the KS equation:
    d_t u(x,t) + d_xx u(x,t) + nu d_xxxx u(x,t) + u(x,t) d_x u(x,t) = 0
    u(x+2pi,t) = u(x,t)
    u(x,0) = init(x)
    using a sine expansion (this is achieved by only considering imaginary wavenumbers.

    The equations are (for v_k = im(u_k))
    d_t v_k = (k^2-nu k^4) v_k + sum(p v_p v_(k-p))

    Input:
        nu: hyperviscosity parameter
        max_wavenumber: Largest wavenumber, corresponds to the nyquist frequency
        init: initial profile.
        tmax: maximal integration time
        dt: time step of output
    Output:
        res: Dictionary of solve_ivp result. In particular,
        res.t contains the time points of the evaluation and
        res.y is a (max_wavenumber, res.t.size) nump array.
    """
    # frequencies
    K = np.arange(1.0, max_wavenumber + 1.0)  # /(2*np.pi)
    g = K**2 - nu * K**4

    def state_tmp(state):
        # make the 2N+1-size vector -fN vN^*, -f(N-1) v(N-1)^*, ..., f(N-1) v(N-1), fN vN
        tmp_w = np.zeros(2 * state.size + 1)
        tmp_w[: state.size] = (K * state)[::-1]
        tmp_w[state.size + 1 :] = K * state

        tmp_v = np.zeros(2 * state.size + 1)
        tmp_v[: state.size] = -state[::-1]
        tmp_v[state.size + 1 :] = state

        return tmp_w, tmp_v

    # State vector is k=1 to k=max_wavenumber
    def KS_system(time, state, dealias=False):
        tmp_w, tmp_v = state_tmp(state)

        if dealias:
            tmp_w = np.pad(tmp_w, state.size)
            tmp_v = np.pad(tmp_v, state.size)

            start = 2 * state.size + 1
            stop = -state.size
        else:
            start = state.size + 1
            stop = None

        res = g * state + convolve(tmp_w, tmp_v, mode="same")[start:stop]

        return res

    # Some useful functions to compute the jacobian
    F = sp.coo_matrix(K)
    F = sp.vstack([F for i in range(max_wavenumber)])

    indices = np.arange(max_wavenumber - 1, -max_wavenumber, -1)

    G = sp.spdiags(g, 0, g.size, g.size, format="coo")

    def KS_jac(time, state):
        tmp = state_tmp(state)[1][1:-1]

        W = sp.diags(tmp, indices, format="coo")

        return (G + F.multiply(W)).toarray()

    return solve_ivp(
        KS_system,
        [0, tmax],
        init,
        method="Radau",
        jac=KS_jac,
        t_eval=np.arange(0, tmax, dt),
        atol=1e-8,
        rtol=1e-8,
    )


def wavenumber_spectrum(res, cutoff):
    """
    Return <u_k u_{-k}> where the average is over time.
    Input is res from solve_ivp (or run_KS_finite_wavenumber)
    cutoff is a cutoff in time to remove the initial transient.
    """

    tmin_idx = res.t[res.t < cutoff].size

    return np.mean(np.real(res.y[:, tmin_idx:] * np.conj(res.y[:, tmin_idx:])), axis=1)

def test_dealias(res):
    max_wavenumber = res.y[:,0].size
    K = np.arange(1.0, max_wavenumber + 1.0)  # /(2*np.pi)

    def state_tmp(state):
        # make the 2N+1-size vector -fN vN^*, -f(N-1) v(N-1)^*, ..., f(N-1) v(N-1), fN vN
        tmp_w = np.zeros(2 * max_wavenumber + 1, dtype="complex")
        tmp_w[: state.size] = -(K * np.conj(state))[::-1]
        tmp_w[state.size + 1 :] = K * state

        tmp_v = np.zeros(2 * max_wavenumber + 1, dtype="complex")
        tmp_v[: state.size] = np.conj(state)[::-1]
        tmp_v[state.size + 1 :] = state

        return tmp_w, tmp_v

    # State vector is k=1 to k=max_wavenumber
    def conv(time, res, dealias=True):
        tmp_w, tmp_v = state_tmp(res.y[:,time])

        if dealias:
            tmp_w = np.pad(tmp_w, max_wavenumber)
            tmp_v = np.pad(tmp_v, max_wavenumber)

            start = 2 * max_wavenumber + 1
            stop = -max_wavenumber

        else:
            start = max_wavenumber + 1
            stop = None

        return convolve(tmp_w, tmp_v, mode="same")[start:stop]
    
    ERR = np.zeros(res.t.size)
    for i in range(res.t.size):
        ERR[i] = np.sum(np.abs(conv(i,res,dealias=True) - conv(i,res,dealias=False)))
    return ERR

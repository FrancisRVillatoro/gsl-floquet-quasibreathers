"""
kg_spectral.py
==============

Fourier pseudospectral discretisation of the nonlinear Klein-Gordon equation

    u_tt - u_xx + F(u) = 0,     x in [-L/2, L/2) periodic (or twisted, see below),

with symplectic splitting in time.  The linear flow of  w_tt - w_xx + kappa w = 0
is solved exactly in Fourier space (a rotation of each mode with frequency
omega_k = sqrt(kappa + k^2)); the remainder  N(w) = F(u) - kappa w  enters as a
momentum kick.  The composition is a symmetric BAB splitting

    Phi_h = B_{b_1 h} A_{a_1 h} B_{b_2 h} ... A_{a_s h} B_{b_{s+1} h},

with the coefficient tables of Blanes and Moan, J. Comput. Appl. Math. 142 (2002)
313-330: 'BM6' is their sixth-order SRKN_11^b (11 force evaluations per step with
first-same-as-last), 'BM4' their fourth-order SRKN_6^b, and 'strang' is the
second-order Stoermer-Verlet/Strang splitting.  All are RKN-type compositions;
the RKN condition [B,[B,[B,A]]] = 0 holds because the kick depends on w only and
the linear flow is Hamiltonian with a quadratic kinetic energy.

Twisted boundary conditions.  Kinks connecting vacua that differ by 2 pi n are
handled with  u = w + s x,  s = 2 pi n / L,  w periodic.  Because F is
2 pi-periodic, F(w + s x) is periodic in x and the equation for w,
w_tt - w_xx + F(w + s x) = 0, is exact (no approximation is made).

Absorbing layer (sponge).  Optional damping  v_t = ... - sigma(x) v  in layers
|x| > x_s of width W; sigma(x) = sigma_0 sin^2(pi xi / 2), xi = (|x| - x_s)/W.
It is applied as the exact map v -> exp(-sigma h) v once per step, split
symmetrically (half at the beginning, half at the end of the step; the halves
merge between consecutive steps).  The kinetic energy removed by each
application is accumulated in `absorbed`, so that  E(t) + absorbed(t)  is
conserved up to the (bounded) energy error of the symplectic scheme.  This
separates radiation from integrator error.

Dealiasing.  With `dealias=True` (default) the Fourier modes with |k| above 2/3
of the Nyquist wavenumber are set to zero in the initial data and in every kick.
For smooth solutions these modes carry no information; removing them also keeps
the step size away from the numerical-resonance bands  h omega_k ~ n pi
(Hairer-Lubich).  Recommended:  h < pi / omega_max  with  omega_max the largest
retained frequency (attribute `h_max`).

Only numpy and scipy are used.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SPLITTING_COEFFICIENTS", "SpectralKG", "sg_breather", "sg_breather_t"]


# --------------------------------------------------------------------------
# Splitting coefficients (BAB compositions).  a: drift/linear-flow weights,
# b: kick weights, len(b) = len(a) + 1.  Sums equal one.
# --------------------------------------------------------------------------

def _bm6():
    # Blanes & Moan (2002), Table 3, SRKN_11^b, order 6, 11 evaluations (FSAL).
    b = [0.0414649985182624, 0.198128671918067, -0.0400061921041533,
         0.0752539843015807, -0.0115113874206879, 0.2366699247869311]
    a = [0.123229775946271, 0.290553797799558, -0.127049212625417,
         -0.246331761062075, 0.357208872795928]
    b_full = b + b[::-1]                       # 12 kicks
    a_full = a + [0.20477705429147] + a[::-1]  # 11 drifts
    return np.array(a_full), np.array(b_full)


def _bm4():
    # Blanes & Moan (2002), Table 3, SRKN_6^b, order 4, 6 evaluations (FSAL).
    b1, b2, b3 = 0.0829844064174052, 0.396309801498368, -0.0390563049223486
    a1, a2 = 0.245298957184271, 0.604872665711080
    a3 = 0.5 - (a1 + a2)
    b4 = 1.0 - 2.0 * (b1 + b2 + b3)
    return np.array([a1, a2, a3, a3, a2, a1]), np.array([b1, b2, b3, b4, b3, b2, b1])


def _strang():
    return np.array([1.0]), np.array([0.5, 0.5])


SPLITTING_COEFFICIENTS = {
    "BM6": dict(order=6, coeffs=_bm6),
    "BM4": dict(order=4, coeffs=_bm4),
    "strang": dict(order=2, coeffs=_strang),
}


# --------------------------------------------------------------------------
# Exact sine-Gordon breather (validation)
# --------------------------------------------------------------------------

def sg_breather(x, t, omega):
    """Exact breather of u_tt - u_xx + sin u = 0, frequency omega < 1."""
    beta = np.sqrt(1.0 - omega * omega)
    return 4.0 * np.arctan(beta / omega * np.sin(omega * t) / np.cosh(beta * x))


def sg_breather_t(x, t, omega):
    """Time derivative of the exact sine-Gordon breather."""
    beta = np.sqrt(1.0 - omega * omega)
    A = beta / omega * np.sin(omega * t) / np.cosh(beta * x)
    At = beta * np.cos(omega * t) / np.cosh(beta * x)
    return 4.0 * At / (1.0 + A * A)


# --------------------------------------------------------------------------
# The integrator
# --------------------------------------------------------------------------

class SpectralKG:
    """Fourier pseudospectral + symplectic splitting integrator for
    u_tt - u_xx + F(u) = 0 on [-L/2, L/2) with periodic or twisted boundary conditions.

    Parameters
    ----------
    L, N     : domain length and number of grid points (N even, preferably a power of 2).
    F        : callable, F(u) vectorised over arrays (the nonlinearity).
    G        : callable, potential with G' = F (only needed for the energy).
    kappa    : linear mass used in the exact linear flow; best set to F'(u_vacuum).
    twist    : integer n; u = w + (2 pi n / L) x with w periodic.
    sponge   : None or dict(x_s=..., width=..., sigma0=...).
    method   : 'BM6' (default), 'BM4' or 'strang'.
    dealias  : zero the top third of the spectrum (2/3 rule).
    G_ref    : value of u at which G is taken as zero for the energy (a vacuum).
    """

    def __init__(self, L, N, F, G=None, kappa=1.0, twist=0, sponge=None,
                 method="BM6", dealias=True, G_ref=0.0):
        self.L, self.N = float(L), int(N)
        self.F, self.G = F, G
        self.kappa = float(kappa)
        self.dx = self.L / self.N
        self.x = -self.L / 2.0 + self.dx * np.arange(self.N)
        self.k = 2.0 * np.pi * np.fft.rfftfreq(self.N, d=self.dx)
        self.omega = np.sqrt(self.kappa + self.k * self.k)
        self.twist = int(twist)
        self.s = 2.0 * np.pi * self.twist / self.L
        self.G_ref = float(G_ref)
        self.method = method
        self.a, self.b = SPLITTING_COEFFICIENTS[method]["coeffs"]()
        self.order = SPLITTING_COEFFICIENTS[method]["order"]
        # dealiasing mask
        kmax = np.max(np.abs(self.k))
        self.mask = (np.abs(self.k) <= (2.0 / 3.0) * kmax) if dealias else np.ones_like(self.k, dtype=bool)
        self.omega_max = float(np.max(self.omega[self.mask]))
        self.h_max = np.pi / self.omega_max
        # sponge
        self.sigma = np.zeros(self.N)
        if sponge is not None:
            xs, W, s0 = float(sponge["x_s"]), float(sponge["width"]), float(sponge["sigma0"])
            xi = np.clip((np.abs(self.x) - xs) / W, 0.0, 1.0)
            self.sigma = s0 * np.sin(0.5 * np.pi * xi) ** 2
        self.has_sponge = bool(np.any(self.sigma > 0.0))
        self.absorbed = 0.0
        self.t = 0.0
        self._cache_h = None
        self.what = None
        self.vhat = None

    # --- state ------------------------------------------------------------------
    def set_initial(self, u0, v0):
        """Set u(x, 0) = u0 and u_t(x, 0) = v0 (arrays on self.x); u0 - s x must be periodic."""
        w0 = np.asarray(u0, dtype=float) - self.s * self.x
        self.what = np.fft.rfft(w0) * self.mask
        self.vhat = np.fft.rfft(np.asarray(v0, dtype=float)) * self.mask
        self.t = 0.0
        self.absorbed = 0.0

    @property
    def w(self):
        return np.fft.irfft(self.what, self.N)

    @property
    def u(self):
        return self.w + self.s * self.x

    @property
    def v(self):
        return np.fft.irfft(self.vhat, self.N)

    # --- energy ----------------------------------------------------------------
    def energy(self, interior=None):
        """Total energy int [v^2/2 + u_x^2/2 + G(u) - G(G_ref)] dx (spectral u_x).
        With interior=(x_a, x_b) only the grid points in that interval are summed."""
        if self.G is None:
            raise ValueError("a potential G is required for the energy")
        w, v = self.w, self.v
        ux = np.fft.irfft(1j * self.k * self.what, self.N) + self.s
        dens = 0.5 * v * v + 0.5 * ux * ux + (self.G(w + self.s * self.x) - self.G(self.G_ref))
        if interior is not None:
            sel = (self.x >= interior[0]) & (self.x <= interior[1])
            return float(np.sum(dens[sel]) * self.dx)
        return float(np.sum(dens) * self.dx)

    # --- building blocks -----------------------------------------------------------
    def _prepare(self, h):
        if self._cache_h == h:
            return
        self._rot = {}
        for a in np.unique(self.a):
            th = a * h * self.omega
            self._rot[a] = (np.cos(th), np.sin(th) / self.omega, -self.omega * np.sin(th))
        self._damp_full = np.exp(-self.sigma * h)
        self._damp_half = np.exp(-0.5 * self.sigma * h)
        self._cache_h = h

    def _drift(self, a):
        c, s_over_w, minus_w_s = self._rot[a]
        what, vhat = self.what, self.vhat
        self.what = c * what + s_over_w * vhat
        self.vhat = minus_w_s * what + c * vhat

    def _kick_hat(self):
        """rfft of the kick N(w) = F(w + s x) - kappa w at the current state; also returns u."""
        w = np.fft.irfft(self.what, self.N)
        u = w + self.s * self.x
        return np.fft.rfft(self.F(u) - self.kappa * w) * self.mask, u

    def _damp(self, factor):
        v = np.fft.irfft(self.vhat, self.N)
        vd = v * factor
        self.absorbed += 0.5 * float(np.sum(v * v - vd * vd)) * self.dx
        self.vhat = np.fft.rfft(vd) * self.mask

    # --- time stepping ---------------------------------------------------------------
    def run(self, h, nsteps, observer=None, every=1):
        """Advance nsteps steps of size h.  observer(self, u) is called at step
        boundaries every `every` steps (u is the physical-space field at that moment)."""
        self._prepare(h)
        a, b = self.a, self.b
        nb = len(b)
        if self.has_sponge:
            self._damp(self._damp_half)
        Nhat, u = self._kick_hat()
        self.vhat = self.vhat - b[0] * h * Nhat
        if observer is not None:
            observer(self, u)
        for n in range(nsteps):
            for i in range(len(a)):
                self._drift(a[i])
                if i < len(a) - 1:
                    Nhat, u = self._kick_hat()
                    self.vhat = self.vhat - b[i + 1] * h * Nhat
            Nhat, u = self._kick_hat()
            self.t += h
            last = (n == nsteps - 1)
            if last:
                self.vhat = self.vhat - b[nb - 1] * h * Nhat
                if self.has_sponge:
                    self._damp(self._damp_half)
            else:
                if self.has_sponge:
                    self.vhat = self.vhat - b[nb - 1] * h * Nhat
                    self._damp(self._damp_full)
                    self.vhat = self.vhat - b[0] * h * Nhat
                else:
                    self.vhat = self.vhat - (b[nb - 1] + b[0]) * h * Nhat
            if observer is not None and ((n + 1) % every == 0):
                observer(self, u)
        return self

    def run_until(self, h, t_end, **kw):
        nsteps = int(round((t_end - self.t) / h))
        return self.run(h, nsteps, **kw)


# --------------------------------------------------------------------------
# Small utilities for measurements
# --------------------------------------------------------------------------

def zero_crossing_frequency(t, y):
    """Mean angular frequency from the upward zero crossings of the signal y(t)
    (linear interpolation between samples)."""
    t, y = np.asarray(t), np.asarray(y)
    idx = np.nonzero((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
    if idx.size < 2:
        return np.nan
    tc = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    return 2.0 * np.pi / np.mean(np.diff(tc))

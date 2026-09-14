"""
qb_newton.py
============

Newton-Fourier (harmonic-balance) search for stationary quasi-breathers of

    u_tt - u_xx + F(u) = 0,   x in [-L/2, L/2) periodic,

for an odd, 2 pi-periodic nonlinearity F.  The ansatz is the one satisfied by
the sine-Gordon breather: odd in t, antiperiodic over half a period, and even
in x,

    u(x, t) = sum_{j=0}^{K-1} U_j(x) sin(k_j omega t),    k_j = 2j + 1,
    U_j(-x) = U_j(x).

Substituting and projecting on sin(k_j omega t) gives the K coupled equations

    R_j = -(k_j omega)^2 U_j - U_j'' + Pi_j[F(u)] = 0,
    Pi_j[g] = (2/M) sum_{m=0}^{M-1} g(x, t_m) sin(k_j omega t_m),  t_m = m T / M,

discretised in x by Fourier collocation (so that U_j'' is diagonal in the
spatial spectrum).  The Jacobian acts as

    (J dU)_j = -(k_j omega)^2 dU_j - dU_j'' + Pi_j[F'(u) du],  du = sum_l dU_l sin(k_l omega t),

and is applied matrix-free; the Newton step is solved with GMRES preconditioned
by the linear operator  m^2 - (k_j omega)^2 + kappa,  which is diagonal in
(harmonic, spatial Fourier) space.  Denominators smaller than `pc_floor` in
modulus are floored, which is what keeps the near-resonant channels (the
harmonics above the mass gap) from destroying the preconditioner.

Two facts to keep in mind when reading the output.

1.  A quasi-breather is not an exact localised solution.  Harmonics with
    k_j omega > sqrt(kappa) lie in the linear continuum, so the converged
    solution carries a standing-wave tail whose amplitude depends on the box
    length L through the near-resonance denominators.  `tail_amplitude`
    reports it; `scan_box_length` shows the dependence.  The physical decay
    rate must be measured in the time domain with absorbing layers
    (`kg_spectral.SpectralKG`), using the Newton solution as initial data.
2.  Imposing evenness in x removes the translation zero mode, so the Jacobian
    is nonsingular apart from the near-resonances above.

Only numpy and scipy are used.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

__all__ = ["HarmonicBalance", "sg_breather_harmonics", "small_amplitude_coefficients",
           "born_radiation", "emitted_harmonics"]


def small_amplitude_coefficients(dressed):
    """(kappa, kappa_3, kappa_5) = (sum m d_m, sum m^3 d_m, sum m^5 d_m) of a DressedGSL.

    F(u) = kappa u - kappa_3 u^3 / 6 + kappa_5 u^5 / 120 + O(u^7).  After the rescaling
    u = sqrt(kappa / kappa_3) w, x, t -> x sqrt(kappa), t sqrt(kappa), the equation agrees
    with sine-Gordon up to cubic order for every nonlinearity with kappa_3 > 0; the first
    deviation is the quintic one, of relative size

        eta_5 = kappa kappa_5 / kappa_3^2 - 1     (zero for sine-Gordon).

    This is a computable indicator of the distance to the integrable equation at small
    amplitude, not a prediction of the radiation, which is beyond all orders in the
    amplitude and depends on the full nonlinearity.
    """
    m, d = dressed.m, dressed.d
    k1 = float(np.sum(m * d))
    k3 = float(np.sum(m ** 3 * d))
    k5 = float(np.sum(m ** 5 * d))
    return k1, k3, k5


def sg_breather_harmonics(x, omega, K, M=None):
    """Odd sine-harmonics U_j(x) of the exact sine-Gordon breather of frequency omega,
    obtained by an FFT in time over one period (M samples)."""
    M = int(M or max(256, 32 * K))
    beta = np.sqrt(1.0 - omega * omega)
    t = 2.0 * np.pi / omega * np.arange(M) / M
    u = 4.0 * np.arctan(beta / omega * np.sin(omega * t)[:, None] / np.cosh(beta * x)[None, :])
    ks = 2 * np.arange(K) + 1
    S = np.sin(2.0 * np.pi * np.outer(np.arange(M), ks) / M)
    return (2.0 / M) * (S.T @ u)


class HarmonicBalance:
    """Harmonic-balance solver for time-periodic, even-in-x solutions.

    Parameters
    ----------
    L, N     : box length and number of collocation points (N even).
    F, dF    : nonlinearity and its derivative, vectorised.
    K        : number of odd harmonics kept (k = 1, 3, ..., 2K-1).
    kappa    : linear mass used by the preconditioner (F'(0) for a vacuum at the origin).
    M        : time samples per period used in the projections (default 16 K).
    pc_floor : floor on |m^2 - (k omega)^2 + kappa| in the preconditioner.
    """

    def __init__(self, L, N, F, dF, K=8, kappa=1.0, M=None, pc_floor=0.25):
        self.L, self.N, self.K = float(L), int(N), int(K)
        self.F, self.dF = F, dF
        self.kappa = float(kappa)
        self.dx = self.L / self.N
        self.x = -self.L / 2.0 + self.dx * np.arange(self.N)
        self.kx = 2.0 * np.pi * np.fft.rfftfreq(self.N, d=self.dx)
        self.k2 = self.kx ** 2
        self.ks = 2 * np.arange(self.K) + 1
        self.M = int(M or 16 * self.K)
        self.S = np.sin(2.0 * np.pi * np.outer(np.arange(self.M), self.ks) / self.M)  # (M, K)
        self.pc_floor = float(pc_floor)
        self._refl = (self.N - np.arange(self.N)) % self.N

    # --- helpers ---------------------------------------------------------------
    def symmetrise(self, U):
        """Project onto functions even in x (kills the translation zero mode)."""
        return 0.5 * (U + U[:, self._refl])

    def to_time(self, U):
        """u(x, t_m) on the M time samples, shape (M, N)."""
        return self.S @ U

    def project(self, g):
        """Sine-harmonic projection of g(x, t_m), shape (M, N) -> (K, N)."""
        return (2.0 / self.M) * (self.S.T @ g)

    def _d2(self, U):
        return np.fft.irfft(-self.k2[None, :] * np.fft.rfft(U, axis=1), self.N, axis=1)

    # --- residual and Jacobian -----------------------------------------------------
    def residual(self, U, omega):
        u = self.to_time(U)
        return (-(self.ks[:, None] * omega) ** 2 * U - self._d2(U) + self.project(self.F(u)))

    def jvp(self, U, omega, dU):
        du = self.to_time(dU)
        Fp = self.dF(self.to_time(U))
        return (-(self.ks[:, None] * omega) ** 2 * dU - self._d2(dU) + self.project(Fp * du))

    def _precond(self, R, omega):
        den = self.k2[None, :] - (self.ks[:, None] * omega) ** 2 + self.kappa
        den = np.where(np.abs(den) < self.pc_floor, np.sign(den + 1e-300) * self.pc_floor, den)
        return np.fft.irfft(np.fft.rfft(R, axis=1) / den, self.N, axis=1)

    # --- Newton ---------------------------------------------------------------------
    def solve(self, U0, omega, tol=1e-11, maxit=40, gmres_tol=1e-4, gmres_restart=60,
              gmres_maxiter=600, verbose=False):
        """Newton-GMRES from the initial guess U0.  Returns (U, info) with
        info = dict(residual, iterations, converged, gmres_iters)."""
        U = self.symmetrise(np.array(U0, dtype=float))
        shape = U.shape
        n = U.size
        hist = []
        for it in range(int(maxit)):
            R = self.symmetrise(self.residual(U, omega))
            rn = np.linalg.norm(R) * np.sqrt(self.dx)
            hist.append(rn)
            if verbose:
                print(f"    Newton {it}: |R| = {rn:.3e}")
            if rn < tol:
                return U, dict(residual=rn, iterations=it, converged=True, history=hist)
            Aop = LinearOperator((n, n), matvec=lambda v: self.symmetrise(
                self.jvp(U, omega, v.reshape(shape))).ravel(), dtype=float)
            Mop = LinearOperator((n, n), matvec=lambda v: self.symmetrise(
                self._precond(v.reshape(shape), omega)).ravel(), dtype=float)
            dU, _ = gmres(Aop, -R.ravel(), M=Mop, rtol=gmres_tol, restart=gmres_restart,
                          maxiter=gmres_maxiter)
            dU = self.symmetrise(dU.reshape(shape))
            # damped step if the full step does not reduce the residual
            lam = 1.0
            for _ in range(8):
                Un = U + lam * dU
                rn_new = np.linalg.norm(self.symmetrise(self.residual(Un, omega))) * np.sqrt(self.dx)
                if rn_new < rn:
                    break
                lam *= 0.5
            U = U + lam * dU
        R = self.symmetrise(self.residual(U, omega))
        rn = np.linalg.norm(R) * np.sqrt(self.dx)
        hist.append(rn)
        return U, dict(residual=rn, iterations=int(maxit), converged=rn < tol, history=hist)

    # --- diagnostics -------------------------------------------------------------------
    def tail_amplitude(self, U, x_core=None):
        """max_j max_{|x| > x_core} |U_j(x)|; the standing-wave tail of the quasi-breather.
        Default x_core is a quarter of the box."""
        x_core = 0.25 * self.L if x_core is None else float(x_core)
        sel = np.abs(self.x) > x_core
        return float(np.max(np.abs(U[:, sel])))

    def core_amplitude(self, U):
        """max over t of |u(0, t)| (peak amplitude at the centre)."""
        i0 = int(np.argmin(np.abs(self.x)))
        return float(np.max(np.abs(self.to_time(U)[:, i0])))

    def initial_data(self, U, omega, t=0.0):
        """(u, u_t) at time t, for use as initial data of a time-domain run."""
        ph = self.ks * omega * t
        u = (np.sin(ph)[None, :] @ U).ravel() if False else np.einsum("j,jn->n", np.sin(ph), U)
        v = np.einsum("j,jn->n", self.ks * omega * np.cos(ph), U)
        return u, v

    def energy(self, U, omega, G, G_ref=0.0, t=0.0):
        """Energy of the field at time t (spectral u_x)."""
        u, v = self.initial_data(U, omega, t)
        ux = np.fft.irfft(1j * self.kx * np.fft.rfft(u), self.N)
        return float(np.sum(0.5 * v * v + 0.5 * ux * ux + (G(u) - G(G_ref))) * self.dx)


def continue_in_omega(hb, U, omega_from, omega_to, n_steps, **kw):
    """Natural continuation in the frequency; returns the list of (omega, U, info)."""
    out = []
    for om in np.linspace(omega_from, omega_to, int(n_steps) + 1)[1:]:
        U, info = hb.solve(U, om, **kw)
        out.append((om, U.copy(), info))
        if not info["converged"]:
            break
    return out


# --------------------------------------------------------------------------
# Radiation estimates
# --------------------------------------------------------------------------

def born_radiation(hb, U, omega, kappa, E0, first_born=True, k_max=9):
    """First-Born (Kivshar-Malomed-type) estimate of the harmonic radiation.

    For a harmonic with k omega > sqrt(kappa) the equation for U_k is
    U_k'' + q_k^2 U_k = N_k, with q_k^2 = (k omega)^2 - kappa and the source
    N_k = Pi_k[F(u) - kappa u].  With outgoing conditions the far field is
    |A_k| = |Nhat_k(q_k)| / (2 q_k),  Nhat_k(q) = int N_k(x) cos(q x) dx,
    and the power radiated to both sides is P_k = |A_k|^2 (k omega) q_k, i.e.

        p_k = P_k T / E0 = pi k |Nhat_k(q_k)|^2 / (2 q_k E0)     per period.

    first_born=True builds the source from the fundamental U_1 alone, which is the
    consistent leading order: U_1 is bound (omega < sqrt(kappa)) and exponentially
    localised, so Nhat_k is then independent of the box (6% between L = 140 and 200).

    first_born=False uses the full solution.  Do NOT use that option with a periodic
    box: the radiating harmonics of the box solution carry a standing tail whose
    component inside the core is set by the near-resonance of the box, and it enters
    Nhat_k(q_k) resonantly, changing it by an order of magnitude and its sign between
    L = 140 and L = 200 (checked).  A meaningful all-orders estimate needs outgoing
    boundary conditions in the harmonic balance, which this module does not implement.

    Returns {k: dict(q, Nhat, A, p)} for the radiating harmonics with k <= k_max.
    """
    if first_born:
        Uw = np.zeros_like(U)
        Uw[0] = U[0]
    else:
        Uw = U
    N = hb.project(hb.F(hb.to_time(Uw))) - kappa * Uw
    out = {}
    for j, k in enumerate(hb.ks):
        if k > k_max:
            break
        w2 = (k * omega) ** 2 - kappa
        if w2 <= 0.0:
            continue
        q = np.sqrt(w2)
        Nhat = float(np.sum(N[j] * np.cos(q * hb.x)) * hb.dx)
        A = abs(Nhat) / (2.0 * q)
        out[int(k)] = dict(q=q, Nhat=Nhat, A=A, p=np.pi * k * Nhat ** 2 / (2.0 * q * E0))
    return out


def emitted_harmonics(kg, omega, x_probe, n_periods, samples_per_period=64, n_fit=None):
    """Amplitudes of the emitted harmonics measured on the field itself.

    Integrates `kg` (a kg_spectral.SpectralKG already carrying initial data) for
    n_periods periods with a step that divides the period exactly, recording u at the
    grid point closest to x_probe, and returns {k: |A_k|} from a Hann-windowed FFT of
    the last n_fit periods (default: the second half).  x_probe must lie between the
    core and the absorbing layer.  The k = 1 line is not radiation but the evanescent
    tail of the breather, and is a useful check: it must equal exp(-beta sqrt(kappa) |x|)
    times the core amplitude.
    """
    T = 2.0 * np.pi / omega
    n_fit = n_fit or n_periods // 2
    ip = int(np.argmin(np.abs(kg.x - x_probe)))
    series = []
    kg.run(T / samples_per_period, int(n_periods * samples_per_period),
           observer=lambda kg, u: series.append(u[ip]), every=1)
    s = np.array(series)[-int(n_fit * samples_per_period):]
    w = np.hanning(s.size)
    S = np.fft.rfft(s * w) * 2.0 / np.sum(w)     # normalised by the coherent gain of the window
    return {k: float(abs(S[int(round(k * n_fit))])) for k in range(1, samples_per_period // 2)}

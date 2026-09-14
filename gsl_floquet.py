"""
gsl_floquet.py
==============

Reference implementation of the Kapitza/Floquet-dressed graphene superlattice
(GSL) nonlinearity, i.e. the nonlinearity of Eq. (26) of

    S. V. Kryuchkov, E. I. Kukhar', D. V. Zav'yalov, Laser Phys. 23, 065902 (2013),

written in the dimensionless variables of Martin-Vergara, Rus & Villatoro
(Chaos Solitons Fractals 151 (2021) 111281; 162 (2022) 112530):

    u_tt - u_xx + F_{b,a}(u) = 0,
    F_{b,a}(u) = (1/2pi) int_0^{2pi} F_b(u + a sin(theta)) dtheta,
    F_b(u)     = sin(u) / sqrt(1 + b^2 (1 - cos u)).

Everything is expressed in two exact representations that are cross-checked
in the accompanying test-suite:

  * direct quadrature over the fast phase theta (trapezoidal rule, spectrally
    accurate because the integrand is analytic and 2pi-periodic in theta);
  * the Fourier-Bessel series F_{b,a}(u) = sum_m c_m(b) J_0(m a) sin(m u),
    obtained from F_b(u) = sum_m c_m(b) sin(m u) and the identity
    (1/2pi) int_0^{2pi} sin(m(u + a sin theta)) dtheta = J_0(m a) sin(m u).

Notation
--------
b   : geometric parameter of the GSL, b = Delta_1 / Delta (b = 0 is sine-Gordon)
a   : dimensionless amplitude of the high-frequency (HF) field,
      a = e E_0 d / (hbar omega_HF)
u   : dimensionless vector potential, u = alpha = e A_z d / (hbar c)
G_b : potential of the undressed GSL, G_b'(u) = F_b(u), G_b(0) = 0
G_{b,a}: dressed potential, G_{b,a}'(u) = F_{b,a}(u)

The effective mass of the 2013 paper, Eq. (22), delta = m_0/m(0), satisfies
    delta(a, b) = (b^2 / 2) * F_{b,a}'(0) = (b^2 / 2) * m_eff^2(a, b),
where m_eff^2 = F_{b,a}'(0) is the squared linear frequency of small
oscillations about u = 0 (the "mass" of the Klein-Gordon equation).

Only numpy and scipy are required.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, interpolate, optimize, special

__all__ = [
    "F_gsl", "G_gsl", "dF_gsl",
    "fourier_coefficients",
    "F_dressed_quad", "G_dressed_quad", "dF_dressed_quad",
    "delta_paper", "effective_mass2", "first_zero_mass2",
    "DressedGSL",
]


# --------------------------------------------------------------------------
# Undressed GSL nonlinearity (closed forms)
# --------------------------------------------------------------------------

def F_gsl(u, b):
    """GSL nonlinearity F_b(u) = sin(u) / sqrt(1 + b^2 (1 - cos u))."""
    u = np.asarray(u, dtype=float)
    return np.sin(u) / np.sqrt(1.0 + b * b * (1.0 - np.cos(u)))


def G_gsl(u, b):
    """GSL potential G_b(u) = 2 (1 - cos u) / (1 + sqrt(1 + b^2 (1 - cos u))).

    Satisfies G_b'(u) = F_b(u), G_b(0) = 0, and G_b -> 1 - cos u as b -> 0.
    This form is numerically stable for all b >= 0 (no cancellation).
    """
    u = np.asarray(u, dtype=float)
    s = 1.0 - np.cos(u)
    return 2.0 * s / (1.0 + np.sqrt(1.0 + b * b * s))


def dF_gsl(u, b):
    """Derivative F_b'(u) = G_b''(u) in closed form.

    F_b'(u) = [4 (1 + b^2) cos u - b^2 (3 + cos 2u)] / [4 (1 + b^2 (1 - cos u))^{3/2}].
    This is the Schrodinger potential V_Sch of the 2021 paper evaluated at u,
    and the integrand of Eq. (22) of Kryuchkov-Kukhar'-Zav'yalov (2013).
    """
    u = np.asarray(u, dtype=float)
    num = 4.0 * (1.0 + b * b) * np.cos(u) - b * b * (3.0 + np.cos(2.0 * u))
    den = 4.0 * (1.0 + b * b * (1.0 - np.cos(u))) ** 1.5
    return num / den


# --------------------------------------------------------------------------
# Fourier (sine) coefficients of F_b
# --------------------------------------------------------------------------

def _next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(int(n), 1))))


def fourier_coefficients(b, tol=1e-15, n_samples=None):
    """Sine coefficients c_m(b), m = 1..M, of F_b(u) = sum_m c_m(b) sin(m u).

    Computed by FFT of F_b on a uniform grid (spectrally accurate since F_b is
    analytic and 2pi-periodic).  The series is truncated at the last m with
    |c_m| > tol * max_m |c_m|.  For large b the function develops a boundary
    layer of width ~ 1/b near u = 0 (mod 2pi), so the number of samples grows
    linearly with b.

    Returns
    -------
    c : ndarray, shape (M,), c[m-1] = c_m(b).
    """
    if n_samples is None:
        n_samples = _next_pow2(max(1024, 128 * (1.0 + b)))
    n = int(n_samples)
    u = 2.0 * np.pi * np.arange(n) / n
    fhat = np.fft.rfft(F_gsl(u, b))
    c = -2.0 * fhat.imag / n            # sine coefficients, m = 0..n/2
    c = c[1:]                           # drop m = 0
    cmax = np.max(np.abs(c))
    keep = np.nonzero(np.abs(c) > tol * cmax)[0]
    M = int(keep[-1]) + 1 if keep.size else 1
    return c[:M].copy()


# --------------------------------------------------------------------------
# Direct quadrature over the fast phase
# --------------------------------------------------------------------------

def _n_theta(a, b, n_theta):
    if n_theta is not None:
        return int(n_theta)
    return _next_pow2(max(512, 64.0 * (1.0 + abs(a) * b)))


def _average_over_phase(func, u, a, b, n_theta):
    """(1/2pi) int_0^{2pi} func(u + a sin theta, b) dtheta by the trapezoidal rule."""
    n = _n_theta(a, b, n_theta)
    theta = 2.0 * np.pi * np.arange(n) / n
    u = np.asarray(u, dtype=float)
    shape = u.shape
    arg = u.reshape(-1, 1) + a * np.sin(theta)[None, :]
    val = func(arg, b).mean(axis=1)
    return val.reshape(shape) if shape else float(val[0])


def F_dressed_quad(u, a, b, n_theta=None):
    """Dressed nonlinearity F_{b,a}(u) by direct quadrature over the HF phase."""
    return _average_over_phase(F_gsl, u, a, b, n_theta)


def G_dressed_quad(u, a, b, n_theta=None, ref=0.0):
    """Dressed potential G_{b,a}(u) - G_{b,a}(ref) by direct quadrature.

    G_{b,a}(u) = (1/2pi) int G_b(u + a sin theta) dtheta; note that G_{b,a}(0)
    is not zero in general (it equals the Kapitza-averaged miniband energy shift),
    hence the reference value is subtracted.
    """
    return (_average_over_phase(G_gsl, u, a, b, n_theta)
            - _average_over_phase(G_gsl, ref, a, b, n_theta))


def dF_dressed_quad(u, a, b, n_theta=None):
    """Derivative F_{b,a}'(u) by direct quadrature of F_b'."""
    return _average_over_phase(dF_gsl, u, a, b, n_theta)


# --------------------------------------------------------------------------
# Effective mass: paper's Eq. (22) and our m_eff^2
# --------------------------------------------------------------------------

def delta_paper(a, b):
    """Inverse effective mass delta = m_0 / m(0) of Eq. (22), Laser Phys. 23, 065902.

    delta = b^2/(16 pi) int_0^{2pi} [4(1+b^2) cos(a sin x) - b^2 (3 + cos(2 a sin x))]
                                     / (1 + b^2 (1 - cos(a sin x)))^{3/2} dx.
    Evaluated with an independent adaptive quadrature (scipy.integrate.quad),
    so that it can serve as a check of the series/trapezoid implementations.
    """
    def integrand(x):
        s = a * np.sin(x)
        num = 4.0 * (1.0 + b * b) * np.cos(s) - b * b * (3.0 + np.cos(2.0 * s))
        den = (1.0 + b * b * (1.0 - np.cos(s))) ** 1.5
        return num / den
    val, _ = integrate.quad(integrand, 0.0, 2.0 * np.pi, limit=400, epsabs=1e-13, epsrel=1e-12)
    return b * b / (16.0 * np.pi) * val


def effective_mass2(a, b, method="series"):
    """Squared linear frequency m_eff^2(a, b) = F_{b,a}'(0) about u = 0.

    method = "series" : sum_m m c_m(b) J_0(m a)
           = "quad"   : trapezoidal average of F_b'(a sin theta)
           = "paper"  : 2 delta / b^2 from Eq. (22) (independent quadrature)
    m_eff^2(0, b) = 1 for every b; m_eff^2 < 0 is the "population inversion"
    regime of the 2013 paper (u = 0 becomes a maximum of the dressed potential).
    """
    if method == "series":
        c = fourier_coefficients(b)
        m = np.arange(1, c.size + 1)
        return float(np.sum(m * c * special.j0(m * a)))
    if method == "quad":
        return float(dF_dressed_quad(0.0, a, b))
    if method == "paper":
        if b == 0.0:
            return float(special.j0(a))
        return 2.0 * delta_paper(a, b) / (b * b)
    raise ValueError("method must be 'series', 'quad' or 'paper'")


def first_zero_mass2(b, a_min=0.5, a_max=4.0, at=0.0):
    """First zero in a of F_{b,a}'(at), the squared linear frequency about u = at.

    at = 0  : a_*(b), turn-over of the dressed potential at u = 0 (tends to the
              first zero of J_0, 2.404826, as b -> 0; equals 2.293 for b = 1);
    at = pi : a_**(b), the value beyond which u = pi becomes the vacuum
              (inverted miniband).  Between a_* and a_** the vacua are +-u*.
    """
    c = fourier_coefficients(b)
    m = np.arange(1, c.size + 1)
    f = lambda a: float(np.sum(m * c * special.j0(m * a) * np.cos(m * at)))
    grid = np.linspace(a_min, a_max, 200)
    vals = np.array([f(x) for x in grid])
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if idx.size == 0:
        raise RuntimeError("no zero of m_eff^2 in [a_min, a_max]")
    i = int(idx[0])
    return optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-13)


# --------------------------------------------------------------------------
# Convenience class: series representation, vacua, static kinks
# --------------------------------------------------------------------------

class DressedGSL:
    """Floquet-dressed GSL nonlinearity for fixed (a, b), series representation.

    F(u)  = sum_m c_m J_0(m a) sin(m u)                (dressed nonlinearity)
    G(u)  = sum_m c_m J_0(m a) (1 - cos(m u)) / m      (dressed potential, G(0) = 0)
    dF(u) = sum_m m c_m J_0(m a) cos(m u)              (derivative / Schrodinger potential)

    The series are absolutely and uniformly convergent (|J_0| <= 1 and c_m decays
    exponentially), so truncation at |c_m| < tol |c_1| is uniform in u and a.
    """

    def __init__(self, a, b, tol=1e-15):
        self.a = float(a)
        self.b = float(b)
        self.c = fourier_coefficients(self.b, tol=tol)
        self.m = np.arange(1, self.c.size + 1, dtype=float)
        self.d = self.c * special.j0(self.m * self.a)   # dressed coefficients d_m

    # --- nonlinearity, potential, derivative -------------------------------
    # Evaluated with Clenshaw's recurrence: one cos and one sin per point and
    # M fused multiply-adds, instead of M transcendental functions per point.
    # For  S = sum_m d_m sin(m u):  b_m = d_m + 2 cos(u) b_{m+1} - b_{m+2},  S = b_1 sin u.
    # For  C = sum_m e_m cos(m u):  same recurrence,  C = b_1 cos u - b_2.

    @staticmethod
    def _clenshaw(coef, u):
        u = np.asarray(u, dtype=float)
        alpha = 2.0 * np.cos(u)
        b1 = np.zeros_like(u)
        b2 = np.zeros_like(u)
        for c in coef[::-1]:
            b1, b2 = c + alpha * b1 - b2, b1
        return b1, b2

    def F(self, u):
        """Dressed nonlinearity F_{b,a}(u) = sum_m d_m sin(m u)."""
        b1, _ = self._clenshaw(self.d, u)
        return b1 * np.sin(np.asarray(u, dtype=float))

    def G(self, u, ref=0.0):
        """Dressed potential normalised so that G(ref) = 0: sum_m (d_m/m)(1 - cos m u)."""
        e = self.d / self.m
        b1, b2 = self._clenshaw(e, u)
        g = np.sum(e) - (b1 * np.cos(np.asarray(u, dtype=float)) - b2)
        if ref != 0.0:
            r1, r2 = self._clenshaw(e, np.array(ref))
            g = g - (np.sum(e) - (r1 * np.cos(ref) - r2))
        return g

    def dF(self, u):
        """Derivative F_{b,a}'(u) = sum_m m d_m cos(m u)."""
        b1, b2 = self._clenshaw(self.m * self.d, u)
        return b1 * np.cos(np.asarray(u, dtype=float)) - b2

    # Direct (outer-product) evaluations, kept for cross-checking Clenshaw.
    def F_direct(self, u):
        u = np.asarray(u, dtype=float)
        return np.sin(np.multiply.outer(u, self.m)) @ self.d

    def G_direct(self, u):
        u = np.asarray(u, dtype=float)
        return (1.0 - np.cos(np.multiply.outer(u, self.m))) @ (self.d / self.m)

    def dF_direct(self, u):
        u = np.asarray(u, dtype=float)
        return np.cos(np.multiply.outer(u, self.m)) @ (self.m * self.d)

    def dG_from(self, u, u_ref):
        """G(u) - G(u_ref) evaluated without cancellation.

        Uses cos(m u_ref) - cos(m u) = 2 sin(m (u + u_ref)/2) sin(m (u - u_ref)/2),
        which stays accurate when u - u_ref is tiny (kink tails).
        """
        u = np.asarray(u, dtype=float)
        sp = np.sin(np.multiply.outer(0.5 * (u + u_ref), self.m))
        sm = np.sin(np.multiply.outer(0.5 * (u - u_ref), self.m))
        return (2.0 * sp * sm) @ (self.d / self.m)

    @property
    def mass2(self):
        """m_eff^2 = F'(0) = sum_m m d_m."""
        return float(np.sum(self.m * self.d))

    # --- critical points of the potential in [0, 2pi) ------------------------
    def critical_points(self, n_grid=4000):
        """Zeros of F in [0, 2pi), classified by the sign of F'.

        Returns two sorted arrays (minima, maxima) of the dressed potential.
        F is odd and 2pi-periodic, so u = 0 and u = pi are always critical.
        """
        u = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
        f = self.F(u)
        zeros = []
        for i in range(n_grid):
            j = (i + 1) % n_grid
            fi, fj = f[i], f[j]
            ui, uj = u[i], (u[j] if j > i else 2.0 * np.pi)
            if fi == 0.0:
                zeros.append(ui)
            elif fi * fj < 0.0:
                zeros.append(optimize.brentq(self.F, ui, uj, xtol=1e-14))
        zeros = np.unique(np.round(np.array(zeros) % (2.0 * np.pi), 12))
        curv = self.dF(zeros)
        minima = zeros[curv > 0.0]
        maxima = zeros[curv < 0.0]
        return minima, maxima

    def vacua(self):
        """Minima of the dressed potential in [0, 2pi) (the vacua)."""
        return self.critical_points()[0]

    @classmethod
    def from_coefficients(cls, d):
        """A nonlinearity F(u) = sum_m d_m sin(m u) with explicit coefficients d = [d_1, d_2, ...]
        (no HF dressing).  Used for the canonical problems sin u + r V_m, V_m = sin(m u) - m sin u,
        i.e. d_1 = 1 - m r, d_m = r."""
        out = cls.__new__(cls)
        out.a, out.b = float("nan"), float("nan")
        out.c = np.asarray(d, dtype=float)
        out.m = np.arange(1, out.c.size + 1, dtype=float)
        out.d = out.c.copy()
        return out

    def shifted(self, u_v):
        """Return the same nonlinearity expanded about the vacuum u_v, in w = u - u_v.

        Only u_v = 0 and u_v = pi are allowed, because those are the only vacua about which
        the shifted nonlinearity is still odd and 2 pi-periodic:
        F(pi + w) = sum_m d_m sin(m pi + m w) = sum_m (-1)^m d_m sin(m w).
        The result is a DressedGSL with the same (a, b) whose F, G, dF, vacua and kinks are
        all expressed in w; this is what regime III (vacuum at u = pi, inverted miniband)
        requires in order to reuse the machinery written for a vacuum at the origin.
        """
        u_v = float(u_v)
        if abs(u_v) < 1e-12:
            sign = np.ones_like(self.m)
        elif abs(abs(u_v) - np.pi) < 1e-12:
            sign = (-1.0) ** self.m
        else:
            raise ValueError("only u_v = 0 and u_v = pi give an odd shifted nonlinearity")
        out = DressedGSL.__new__(DressedGSL)
        out.a, out.b = self.a, self.b
        out.c, out.m = self.c, self.m
        out.d = sign * self.d
        out.u_v = u_v
        return out

    def regime(self):
        """Classify the vacuum structure from the curvature at u = 0 and u = pi.

        'I'   : F'(0) > 0 and F'(pi) < 0, vacuum u = 0 (mod 2pi), barrier at pi.
        'II'  : F'(0) < 0 and F'(pi) < 0, two symmetric vacua +-u* (mod 2pi);
                0 and pi are barrier tops.  At leading order in b this is
                sine-Gordon in w = 2u.
        'III' : F'(0) < 0 and F'(pi) > 0, vacuum u = pi (mod 2pi), inverted
                miniband; a 2pi-kink connects pi and 3pi through 2pi.
        'IV'  : F'(0) > 0 and F'(pi) > 0, two non-degenerate vacua 0 and pi.
        Higher harmonics can in principle add further minima; use vacua().
        """
        k0, kp = float(self.dF(0.0)), float(self.dF(np.pi))
        if k0 > 0 and kp < 0:
            return "I"
        if k0 < 0 and kp < 0:
            return "II"
        if k0 < 0 and kp > 0:
            return "III"
        return "IV"

    # --- static kinks ----------------------------------------------------------
    def static_kink(self, u1, u2, n_side=1500, s_max=18.0, gl_order=12):
        """Static kink connecting two adjacent degenerate vacua u1 < u2.

        Uses the first integral u_x^2 / 2 = G(u) - G(u1) and the quadrature
        x(u) = int_{u_c}^{u} dv / sqrt(2 (G(v) - G(u1))), where u_c is the barrier
        top in (u1, u2).  On each side the substitution w = |u_c - u_vac| e^{-s}
        (w = distance to the vacuum) removes the logarithmic singularity; the
        integrand w / sqrt(2 dG) is smooth in s and tends to 1/mu, so a composite
        Gauss-Legendre rule (gl_order nodes on n_side panels) is accurate to
        near machine precision.  G(u) - G(u1) is evaluated with dG_from, which
        has no cancellation in the tails.

        Returns
        -------
        x, u : arrays with x increasing and u(x) increasing from u1 to u2,
               centred so that u(0) = u_c.  The tails reach w ~ e^{-s_max}
               (1.5e-8 for the default s_max = 18); s_max is limited by the
               accuracy of the vacuum location (|F(u1)| ~ 1e-15), beyond which
               the first integral is no longer positive definite.
        """
        u1, u2 = float(u1), float(u2)
        if abs(float(self.dG_from(u2, u1))) > 1e-10:
            raise ValueError("vacua are not degenerate; no static kink exists")
        _, maxima = self.critical_points()
        cands = np.array([c for c in np.concatenate([maxima - 2 * np.pi, maxima, maxima + 2 * np.pi])
                          if u1 < c < u2])
        if cands.size != 1:
            raise ValueError("expected exactly one barrier top between the vacua")
        uc = float(cands[0])
        nodes, weights = np.polynomial.legendre.leggauss(int(gl_order))

        def side(u_vac, sign):
            # sign = -1: lower side (u from u_c down to u1, x < 0)
            # sign = +1: upper side (u from u_c up to u2, x > 0)
            s_edges = np.linspace(0.0, s_max, int(n_side) + 1)
            h = s_edges[1] - s_edges[0]
            s_nodes = (s_edges[:-1, None] + 0.5 * h * (nodes[None, :] + 1.0)).ravel()
            w_nodes = abs(uc - u_vac) * np.exp(-s_nodes)
            uu_nodes = u_vac - sign * w_nodes
            dG = self.dG_from(uu_nodes, u1)
            if np.any(dG <= 0.0):
                raise RuntimeError("G(u) - G(u1) not positive along the orbit; reduce s_max "
                                   "or refine the vacuum location")
            integrand = w_nodes / np.sqrt(2.0 * dG)
            panel = (integrand.reshape(int(n_side), -1) * weights[None, :]).sum(axis=1) * 0.5 * h
            xs = np.concatenate([[0.0], np.cumsum(panel)])      # |x| at s_edges
            uu = u_vac - sign * abs(uc - u_vac) * np.exp(-s_edges)
            return uu, sign * xs

        u_lo, x_lo = side(u1, -1.0)
        u_hi, x_hi = side(u2, +1.0)
        x = np.concatenate([x_lo[::-1], x_hi[1:]])
        u = np.concatenate([u_lo[::-1], u_hi[1:]])
        return x, u

    def kink_energy(self, u1, u2):
        """Energy of the static kink, E = int_{u1}^{u2} sqrt(2 (G(u) - G(u1))) du."""
        f = lambda v: np.sqrt(max(2.0 * float(self.dG_from(v, u1)), 0.0))
        val, _ = integrate.quad(f, u1, u2, limit=200)
        return val

    def kink_on_grid(self, u1, u2, x, **kw):
        """Static kink evaluated on a user grid x by cubic-spline interpolation
        of the quadrature profile (interpolation error ~1e-8 with the defaults);
        outside the computed range the vacuum values are used."""
        xk, uk = self.static_kink(u1, u2, **kw)
        spl = interpolate.CubicSpline(xk, uk)
        x = np.asarray(x, dtype=float)
        out = spl(np.clip(x, xk[0], xk[-1]))
        out = np.where(x < xk[0], u1, out)
        out = np.where(x > xk[-1], u2, out)
        return out

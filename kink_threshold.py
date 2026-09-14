"""
kink_threshold.py -- existence of the kink internal mode by threshold shooting.

For the linearised operator L = -d^2/dx^2 + F'(u_k(x)) about the static kink, the internal
mode is the lowest odd eigenstate.  At the continuum edge E = kappa the odd solution of
L psi = kappa psi with psi(0) = 0, psi'(0) = 1 tends to alpha + beta x; by Sturm's
oscillation theorem the internal mode exists iff beta < 0 (the edge solution has a node),
and it merges into the continuum exactly where beta = 0.  This is exact and free of box-size
limitations, unlike diagonalisation, which loses the mode when its decay length exceeds the box.
"""
import numpy as np
from scipy import integrate, interpolate, optimize, special
import gsl_floquet as gf


def edge_slope(d, X=None):
    """beta = psi'(X) for the odd edge solution on the 2pi-kink of the nonlinearity d."""
    kappa = float(d.mass2)
    X = X or 60.0 / np.sqrt(kappa)
    xk, uk = d.static_kink(0.0, 2 * np.pi)
    spl = interpolate.CubicSpline(xk, uk)

    def V(x):
        u = spl(np.clip(x, xk[0], xk[-1]))
        return float(d.dF(u)) - kappa

    sol = integrate.solve_ivp(lambda x, y: [y[1], V(x) * y[0]], (0.0, X), [0.0, 1.0],
                              method="DOP853", rtol=1e-11, atol=1e-13)
    return sol.y[1, -1]


def a_internal_mode_vanishes(b, a_lo=0.5, a_hi=None):
    """a_k(b): the HF amplitude at which the internal mode of the GSL kink reaches the edge."""
    a_hi = a_hi or gf.first_zero_mass2(b) - 0.05
    f = lambda a: edge_slope(gf.DressedGSL(a, b))
    return optimize.brentq(f, a_lo, a_hi, xtol=1e-6)


def canonical_slope_derivative(m, r=1e-3):
    """d beta / d r at r = 0 for sin u + r V_m (central difference); beta(0) = 0 for sine-Gordon."""
    def dcoef(rr):
        d = np.zeros(max(m, 2)); d[0] = 1.0 - m * rr; d[m - 1] += rr
        return gf.DressedGSL.from_coefficients(d)
    return (edge_slope(dcoef(r)) - edge_slope(dcoef(-r))) / (2 * r)


if __name__ == "__main__":
    j01_2 = special.jn_zeros(0, 1)[0] / 2
    print("sine-Gordon check: beta(0) = %.2e (threshold state tanh x, must vanish)"
          % edge_slope(gf.DressedGSL(0.0, 0.0)))
    print("GSL at a = 0: beta = %.4f (b = 0.5), %.4f (b = 1.0)  [negative: internal mode present]"
          % (edge_slope(gf.DressedGSL(0.0, 0.5)), edge_slope(gf.DressedGSL(0.0, 1.0))))
    w = {m: canonical_slope_derivative(m) for m in [2, 3, 4]}
    print("first-order threshold slopes d beta/d r for V_2, V_3, V_4: %s" % {m: round(v, 5) for m, v in w.items()})
    print("   ratios w_3/w_2 = %.4f,  w_4/w_2 = %.4f" % (w[3] / w[2], w[4] / w[2]))
    a0 = j01_2
    print("\n b       a_k measured   a_k predicted (V_3 + V_4 terms)   shift/b^2 measured   predicted")
    for b in [0.125, 0.25, 0.5, 1.0]:
        ak = a_internal_mode_vanishes(b)
        c = gf.fourier_coefficients(b)
        den = 2 * c[1] * special.j1(2 * a0)
        pred = a0 + (c[2] * special.j0(3 * a0) * w[3] / w[2] + c[3] * special.j0(4 * a0) * w[4] / w[2]) / den
        print("  %.3f    %.5f         %.5f                        %+.4f              %+.4f"
              % (b, ak, pred, (ak - a0) / b ** 2, (pred - a0) / b ** 2))
    print("\n j_01/2 = %.5f" % j01_2)

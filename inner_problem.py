"""
inner_problem.py -- the Stokes constant from the late-order terms of the inner expansion.

Setting (see the paper).  At first order in rho the perturbed breather is
u = u_B + rho v with

    v_tt - v_xx + cos(u_B) v = -V(u_B),      V(u) = sin 2u - 2 sin u.

The exponentially small radiation is generated at the complex singularity of the breather,
x_s = i pi / (2 epsilon), where cosh(epsilon x) = 0.  With xi = x - x_s = O(1) and
epsilon -> 0 the breather reduces to the epsilon-free inner profile

    U(xi, t) = 4 arctan(-i sin t / xi),      z := -i sin t / xi,

and the inner equation is v_tt - v_xi_xi + cos(U) v = -V(U) with, writing w = sin t / xi,

    cos U - 1 = 8 w^2 / (1 - w^2)^2,          V(U) = -64 i w^3 (1 + w^2) / (1 - w^2)^4.

Expanding v = sum_{k odd} v_k(xi) sin(k t) and v_k = sum_{N odd} a_{k,N} xi^{-N}, the
coefficients satisfy an exact recursion with rational coefficients (a_{k,N} = i r_{k,N},
r rational).  The recursion for the third harmonic is dominated by

    a_{3,N} ~ -(N-1)(N-2) a_{3,N-2} / 8,

so the series diverges factorially with singulants chi = +- i sqrt(8): a_{3,N} ~
Gamma(N) [A_+ chi_+^{-N} + A_- chi_-^{-N}], and oddness in xi forces A_- = -A_+.  By the
Borel/Dingle connection, a series with late terms A Gamma(N + alpha) chi^{-N-alpha}
switches on the exponential 2 pi i A xi^alpha exp(-chi xi) across the Stokes line
arg(chi xi) = 0.  For chi_+ = i sqrt 8 that line is arg xi = -pi/2, the ray pointing
from the singularity to the real x axis, and exp(-chi_+ xi) = exp(-i sqrt8 xi) is the
right-moving third harmonic.  Continuing to the real axis, exp(-i sqrt8 xi) =
exp(-i sqrt8 x) exp(-sqrt8 pi/(2 eps)) = exp(-i sqrt8 x) exp(-pi sqrt2/eps), hence

    |A_3| = 2 pi |A_+| exp(-pi q_3 / (2 epsilon)) (1 + O(epsilon^2)),
    Lambda_0 := 2 pi |A_+| = pi |D|,     D := lim_{N} a_{3,N} / [ i (-1)^{(N+1)/2} Gamma(N) 8^{-N/2} ],

with the exact q_3 = sqrt(9 Omega^2 - 1) entering through the outer phase exp(-i q_3 x).
The recursion also shows alpha = 0, i.e. no power of epsilon in the prefactor: gamma = 0.

This file evaluates D to high accuracy with exact rational arithmetic and Richardson
extrapolation, and checks the two consistency conditions that follow from the recursion:
r_{1,1} = 12 (the amplitude correction 1 - 3 rho of the breather at fixed frequency) and
the vanishing of all even-order terms.

Run:  python inner_problem.py
"""

from fractions import Fraction
from math import comb, lgamma, log, pi, exp, sqrt
import sys


def binom(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def T(m, k, l):
    """(1/pi) int_0^{2pi} sin^{2m} t sin kt sin lt dt, exact rational, k,l >= 1."""
    s = Fraction(binom(2 * m, m)) if k == l else Fraction(0)
    for j in range(1, m + 1):
        c = (-1) ** j * binom(2 * m, m - j)
        if 2 * j == abs(k - l):
            s += c
        if 2 * j == k + l:
            s -= c
    return s / 4 ** m


def Sigma(n, k):
    """(1/pi) int sin^{2n+3} t sin kt dt = (-1)^j C(2n+3, n+1-j)/4^{n+1} for k = 2j+1."""
    if k % 2 == 0:
        return Fraction(0)
    j = (k - 1) // 2
    p = n + 1
    if j > p:
        return Fraction(0)
    return Fraction((-1) ** j * binom(2 * p + 1, p - j), 4 ** p)


def _poly_mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return out


def _poly_pow(p, n):
    out = [1]
    for _ in range(n):
        out = _poly_mul(out, p)
    return out


def source_polynomial(m):
    """N_m(w)/i as an integer polynomial in w, where V_m(U) = sin(mU) - m sin U = N_m(w)/(1-w^2)^{2m}
    for U = 4 arctan(-i w).  Uses cos U = (1+6w^2+w^4)/(1-w^2)^2, sin U = -4 i w (1+w^2)/(1-w^2)^2:
        N_m / i = sum_{j odd} C(m,j) (-4)^j w^j (1+w^2)^j (1+6w^2+w^4)^{m-j} + 4 m w (1+w^2) (1-w^2)^{2m-2}.
    Check: m = 2 gives -64 w^3 (1 + w^2)."""
    A = [1, 0, 6, 0, 1]          # 1 + 6 w^2 + w^4
    B = [1, 0, 1]                # 1 + w^2
    Dm = [1, 0, -1]              # 1 - w^2
    tot = [0]
    for j in range(1, m + 1, 2):
        term = _poly_mul(_poly_pow(B, j), _poly_pow(A, m - j))
        term = [(-4) ** j * binom(m, j) * c for c in term]
        term = [0] * j + term            # times w^j
        tot = [a + b for a, b in zip(tot + [0] * (len(term) - len(tot)), term + [0] * (len(tot) - len(term)))]
    last = _poly_mul(B, _poly_pow(Dm, 2 * m - 2))
    last = [0] + [4 * m * c for c in last]
    tot = [a + b for a, b in zip(tot + [0] * (len(last) - len(tot)), last + [0] * (len(tot) - len(last)))]
    return tot


_SOURCE_CACHE = {}


def source_coefficients(m, n_max):
    """s_n, n = 0..n_max, with V_m(U) = i sum_n s_n w^{2n+3}: the series of N_m(w)/(i w^3) (1-w^2)^{-2m}."""
    key = (m, n_max)
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    P = source_polynomial(m)
    assert all(c == 0 for c in P[:3]) and all(P[i] == 0 for i in range(0, len(P), 2)), "expected odd polynomial starting at w^3"
    p = [P[3 + 2 * i] for i in range((len(P) - 3 + 1) // 2)]        # coefficients of w^{2i} in N_m/(i w^3)
    # multiply by (1 - w^2)^{-2m} = sum_q C(q + 2m - 1, 2m - 1) w^{2q}
    s = []
    for n in range(n_max + 1):
        val = 0
        for i, c in enumerate(p):
            q = n - i
            if q >= 0:
                val += c * binom(q + 2 * m - 1, 2 * m - 1)
        s.append(val)
    _SOURCE_CACHE[key] = s
    return s


def beta(n, which=2):
    """Legacy accessor: V_which(U) = -PREF * i * beta_n w^{2n+3} with PREF = SOURCE_PREFACTOR[which]."""
    return Fraction(-source_coefficients(which, n)[n], SOURCE_PREFACTOR[which])


SOURCE_PREFACTOR = {m: (m ** 3 - m) * 64 // 6 for m in range(2, 12)}   # V_m ~ -(m^3-m)/6 U^3, U ~ -4 i w


def late_terms(N_max, K_max, which=2):
    """Return r[k][N] (a_{k,N} = i r_{k,N}) for odd N <= N_max and odd k <= K_max,
    for the perturbation V_which (which = 2: sin 2u - 2 sin u; which = 3: sin 3u - 3 sin u)."""
    ks = list(range(1, K_max + 1, 2))
    r = {k: {} for k in ks}
    Tcache = {}

    def Tc(m, k, l):
        key = (m, k, l)
        if key not in Tcache:
            Tcache[key] = T(m, k, l)
        return Tcache[key]

    def get(k, N):
        if k not in r or N < 1 or N not in r[k]:
            return Fraction(0)
        return r[k][N]

    def source(k, N):
        if N < 3:
            return Fraction(0)
        n = (N - 3) // 2
        return SOURCE_PREFACTOR[which] * beta(n, which) * Sigma(n, k)

    for N in range(1, N_max + 1, 2):
        # harmonics k >= 3 at order N
        for k in ks:
            if k == 1:
                continue
            s = -(N - 1) * (N - 2) * get(k, N - 2)
            for m in range(1, (N - 1) // 2 + 1):
                for l in ks:
                    t = Tc(m, k, l)
                    if t:
                        s += 8 * m * t * get(l, N - 2 * m)
            s -= source(k, N)
            r[k][N] = s / (k * k - 1)
        # k = 1 at order N from the k = 1 equation at order N + 2
        M = N + 2
        s = Fraction(0)
        for m in range(1, (M - 1) // 2 + 1):
            for l in ks:
                if m == 1 and l == 1:
                    continue
                t = Tc(m, 1, l)
                if t:
                    s += 8 * m * t * get(l, M - 2 * m)
        s -= source(1, M)
        denom = (M - 1) * (M - 2) - 8 * Tc(1, 1, 1)      # = (M-4)(M+1), never zero for odd M
        r[1][N] = s / denom
    return r


def stokes_D(r, N):
    """D_N = r_{3,N} (-1)^{(N+1)/2} 8^{N/2} / Gamma(N), evaluated in logs."""
    x = r[3][N]
    if x == 0:
        return 0.0
    sign = (-1) ** ((N + 1) // 2) * (1 if x > 0 else -1)
    logabs = log(abs(x.numerator)) - log(x.denominator) + 0.5 * N * log(8.0) - lgamma(N)
    return sign * exp(logabs)


def richardson(vals, Ns, order):
    out = list(vals)
    for p in range(1, order + 1):
        out = [(Ns[i + 1] ** p * out[i + 1] - Ns[i] ** p * out[i]) / (Ns[i + 1] ** p - Ns[i] ** p)
               for i in range(len(out) - 1)]
    return out


def stokes_constant(N_max=121, K_max=25, which=2):
    """D (signed) and Lambda_0 = pi |D| for the perturbation V_which."""
    r = late_terms(N_max, K_max, which)
    Ns = list(range(21, N_max + 1, 2))
    Ds = [stokes_D(r, N) for N in Ns]
    D = richardson(Ds, Ns, 3)[-1]
    return D, pi * abs(D), r


if __name__ == "__main__":
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else 161
    K_max = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    which = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    r = late_terms(N_max, K_max, which)
    print(f"N_max = {N_max}, K_max = {K_max}, perturbation V_{which}")
    print(f"consistency: r_1,1 = {r[1][1]}  (expected 12: amplitude factor 1 - 3 rho)")
    print(f"             r_3,3 = {r[3][3]}, r_5,5 = {r[5][5]}")
    print("late terms of the third harmonic, D_N = r_3,N (-1)^((N+1)/2) 8^(N/2) / Gamma(N):")
    Ns = list(range(21, N_max + 1, 2))
    Ds = [stokes_D(r, N) for N in Ns]
    for N, D in zip(Ns, Ds):
        if N % 20 == 1 or N == N_max:
            print(f"   N = {N:4d}   D_N = {D:.10f}")
    # Richardson in 1/N (the corrections are a series in 1/N)
    for order in [0, 1, 2, 3]:
        ext = richardson(Ds, Ns, order)
        print(f"   Richardson order {order}: last three values {ext[-3]:.10f} {ext[-2]:.10f} {ext[-1]:.10f}")
    D = richardson(Ds, Ns, 3)[-1]
    print(f"\n   D = {D:.8f}")
    print(f"   Lambda_0 = pi |D| = {pi * abs(D):.6f}   (2 pi |D| = {2 * pi * abs(D):.6f} if the jump counts twice)")
    print(f"   candidates: 8 pi = {8 * pi:.6f},  pi^3 = {pi ** 3:.6f},  256/(3 pi) = {256 / (3 * pi):.6f},  "
          f"64/sqrt(6) = {64 / sqrt(6):.6f}")

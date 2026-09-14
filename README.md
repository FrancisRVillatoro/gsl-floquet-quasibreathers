# Quenching the radiation of quasi-breathers in a Floquet-dressed graphene superlattice equation

Code, data and figures for the paper by F. Martin-Vergara, F. Rus and F. R. Villatoro.

The graphene superlattice (GSL) equation under a high-frequency field along the superlattice
axis is, after averaging over the fast field, the local Hamiltonian Klein-Gordon equation

    u_tt - u_xx + F(u) = 0,    F(u) = sum_m c_m(b) J_0(m a) sin(m u),

with `b` the miniband ratio and `a` the dimensionless HF amplitude. This repository contains
everything needed to reproduce the study of the radiation of its quasi-breathers and of its
cancellation at the zeros of J_0(2a).

## Layout

    *.py                 library modules and experiment scripts (see below)
    test_*.py            five test-suites, 199 tests, about 45 s in total
    results/*.csv        every measured dataset, one file per quantity
    figures/*.pdf,*.png  the six publication figures, built from results/
    run_all.sh           regenerates every number and figure (about 75 min)
    requirements.txt     pinned library versions

### Library

| file | contents |
|---|---|
| `gsl_floquet.py` | the dressed nonlinearity: Fourier-Bessel coefficients, potential, effective mass, regimes, static kinks, expansion about u = pi |
| `kg_spectral.py` | Fourier pseudospectral integrator with exact linear flow and the sixth-order symmetric RKN splitting of Blanes and Moan; twisted boundary conditions; absorbing layers with exact energy bookkeeping |
| `qb_newton.py` | harmonic-balance Newton-GMRES for stationary quasi-breathers; radiation diagnostics |
| `inner_problem.py` | late-order terms of the inner expansion at the complex singularity, in exact rational arithmetic; Stokes constants |
| `kink_threshold.py` | threshold shooting for the kink internal mode |
| `paper_style.py`, `make_figures.py` | figure style and the six publication figures |

### Experiments

`experiment_H1.py`, `experiment_ac.py`, `experiment_asymptotics.py`, `stokes_constant.py`,
`experiment_ratio.py`, `experiment_second_zero.py`, `experiment_ac_map.py`,
`experiment_kink_modes.py`, `experiment_collisions.py`, plus `example_radiation.py` and
`fig_gsl_floquet.py` as illustrations. Each has a docstring stating what it measures and how
long it takes, and each writes its results to the log and (where applicable) to `results/`.

## Reproducing

    pip install -r requirements.txt
    ./run_all.sh

To redraw the figures without re-running the simulations:

    python make_figures.py

## Data

`results/` holds the measured quantities as CSV, one file per quantity, with the parameters
in the file name. Among them: the radiated power against the HF amplitude and against the
frequency, the harmonic content of the emitted field, the Stokes function K(epsilon), the
cancellation points at both zeros of J_0(2a) and their map in (b, Omega), the continuation at
b = 1, the Stokes constants from the inner problem, the kink spectra and the critical
velocities for capture.

## Numerical conventions

Dimensionless variables of Martin-Vergara, Rus and Villatoro (2021, 2022): t and x scaled by
the plasma frequency of the miniband, u the dimensionless vector potential; `a = e E_0 d /
(hbar omega_HF)`; `b = Delta_1 / Delta`. Quasi-breathers are compared at fixed reduced
frequency `Omega = omega_B / sqrt(kappa)` with `kappa = F'(u_vac)`, which is what makes
different nonlinearities comparable. Radiated power is reported per period, relative to the
total energy.

## Citing

See `CITATION.cff`. Please cite both the archived release and the paper.

## License

BSD 3-Clause, see `LICENSE`.

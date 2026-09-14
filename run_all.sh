#!/usr/bin/env bash
# Regenerate every number and figure of the paper from scratch.
# Times measured on a single core of the machine used for the paper; total about 75 minutes.
set -e

echo "== test-suites (about 45 s) =="
python -m pytest -q test_gsl_floquet.py test_kg_spectral.py test_qb_newton.py \
                   test_inner_problem.py test_kink_threshold.py

echo "== analytic parts (seconds) =="
python inner_problem.py 161 21 2          # Stokes constant of V_2   (15 s)
python inner_problem.py 161 21 3          # Stokes constant of V_3   (15 s)
python kink_threshold.py                  # a_k(b) and the threshold slopes (40 s)

echo "== simulations (the expensive part) =="
python experiment_H1.py                   # radiated power vs a                     ( 6 min)
python experiment_ac.py                   # fine scan, harmonics, first-Born control ( 5 min)
python experiment_asymptotics.py          # exponent, rho^2 law, a_c(b)              (12 min)
python stokes_constant.py                 # K(epsilon) and the GSL prediction        ( 8 min)
python experiment_ratio.py                # K_3/K_2, K_4/K_2 and the b^4 term        ( 4 min)
python experiment_second_zero.py          # second cancellation, regime III          ( 6 min)
python experiment_ac_map.py               # a_c(b, Omega) and b = 1                  (25 min)
python experiment_kink_modes.py           # kink spectrum vs a                       ( 3 min)
python experiment_collisions.py           # critical velocity at b = 1               (20 min)

echo "== publication figures from results/ (30 s) =="
python make_figures.py

echo "done"

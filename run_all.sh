#!/usr/bin/env bash
# Regenerate all archived numerical datasets and publication figures from scratch.
# Results and figures are staged and audited before replacing the checked-in snapshot.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
STAGE=$(mktemp -d "$ROOT/.regen.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
export GSL_RESULTS_DIR="$STAGE/results"
export GSL_FIGURES_DIR="$STAGE/figures"
mkdir -p "$GSL_RESULTS_DIR" "$GSL_FIGURES_DIR"

echo "== test suites =="
python -m pytest -q test_gsl_floquet.py test_kg_spectral.py test_qb_newton.py \
                   test_inner_problem.py test_kink_threshold.py

echo "== analytic threshold calculation =="
python kink_threshold.py

echo "== quasi-breather radiation datasets =="
python experiment_H1.py
python experiment_ac.py
python experiment_asymptotics.py
python stokes_constant.py
python experiment_ratio.py
python experiment_second_zero.py
python experiment_ac_map.py

echo "== kink datasets =="
python experiment_kink_modes.py
python experiment_collisions.py

echo "== targeted convergence closure =="
python experiment_convergence_minimum.py

echo "== audit staged result set =="
python verify_results.py "$GSL_RESULTS_DIR"

echo "== publication figures from staged results =="
python make_figures.py
for stem in fig1_dressed_nonlinearity fig2_cancellation fig3_asymptotics \
            fig4_stokes_and_map fig5_second_zero fig6_kink; do
  test -s "$GSL_FIGURES_DIR/$stem.pdf"
  test -s "$GSL_FIGURES_DIR/$stem.png"
done

echo "== install audited results and figures =="
rm -rf results.previous figures.previous
if [ -d results ]; then mv results results.previous; fi
if [ -d figures ]; then mv figures figures.previous; fi
mv "$GSL_RESULTS_DIR" results
mv "$GSL_FIGURES_DIR" figures
rm -rf results.previous figures.previous
unset GSL_RESULTS_DIR GSL_FIGURES_DIR

echo "done: all archived CSV datasets and publication figures regenerated"

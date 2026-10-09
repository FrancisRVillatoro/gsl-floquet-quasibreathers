"""Verify that a full reproducibility run produced every published numeric dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

EXPECTED = {
    "ac_first_zero_vs_b_Om0.8.csv": ("b","a_c_measured","a_c_predicted_V3_V4"),
    "ac_map_b_Omega.csv": ("b","Omega","a_c"),
    "ac_second_zero_vs_b_Om0.8.csv": ("b","a_c_measured","a_c_predicted_V3_V4"),
    "b1_continuation_Om0.92.csv": ("a","kappa","core_over_pi","A3","A5"),
    "collisions_vcr_b1_Om0.8.csv": ("a","v_cr"),
    "emitted_harmonics_b0.5_Om0.8.csv": ("a","A1_evanescent","A3","A5","A7","flux_per_period_over_E0","sponge_per_period_over_E0"),
    "fine_scan_b0.5_Om0.8.csv": ("a","P_per_period_over_E0","first_born_p3"),
    "gsl_prediction_vs_measurement.csv": ("b","a","rho","A3_predicted","A3_measured"),
    "kink_a_k_vs_b.csv": ("b","a_k_measured","a_k_predicted_V3_V4"),
    "kink_modes_b0.25.csv": ("a","kappa","zero_mode","internal_omega2"),
    "kink_modes_b0.5.csv": ("a","kappa","zero_mode","internal_omega2"),
    "kink_modes_b1.0.csv": ("a","kappa","zero_mode","internal_omega2"),
    "power_vs_Omega_a0.6_b0.5.csv": ("Omega","epsilon","P_per_period_over_E0"),
    "power_vs_a_b0.5_Om0.8.csv": ("a","P_per_period_over_E0"),
    "power_vs_b_a0.6_Om0.8.csv": ("b","rho_exact","P_per_period_over_E0"),
    "second_zero_scans_Om0.8.csv": ("b","a","A3"),
    "stokes_K_of_epsilon.csv": ("Omega","epsilon","K_richardson"),
    "stokes_constants_inner.csv": ("m","D","Lambda0","ratio_to_m2_eps0","ratio_to_m2_eps0.6"),
    "convergence_minimum_a1p25.csv": ("case","K","L","N","h","x_s","width","sigma0","P_per_period_over_E0","hb_residual","energy_balance_relative"),
    "convergence_ac_strict.csv": ("a","P_per_period_over_E0","sqrtP","hb_residual","energy_balance_relative"),
}
OPTIONAL_CLOSURE = set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="results")
    args = ap.parse_args()
    root = Path(args.directory)
    errors=[]
    for name, cols in EXPECTED.items():
        p=root/name
        if not p.is_file():
            errors.append(f"missing {name}")
            continue
        first=p.read_text().splitlines()[0].strip().split(",")
        if tuple(first) != cols:
            errors.append(f"{name}: header {first!r} != {cols!r}")
            continue
        d=np.genfromtxt(p, delimiter=",", names=True)
        if np.size(d)==0:
            errors.append(f"{name}: no rows")
    extra={p.name for p in root.glob('*.csv')} - set(EXPECTED) - OPTIONAL_CLOSURE
    if extra:
        errors.append("unexpected CSV: " + ", ".join(sorted(extra)))
    if errors:
        raise SystemExit("RESULT AUDIT FAILED\n  " + "\n  ".join(errors))
    print(f"RESULT AUDIT PASS: {len(EXPECTED)} archived CSV datasets present with expected schemas")
    for name in sorted(OPTIONAL_CLOSURE):
        if (root/name).is_file():
            print(f"  closure dataset present: {name}")

if __name__ == "__main__":
    main()

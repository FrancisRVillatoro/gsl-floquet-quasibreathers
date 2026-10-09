"""Small deterministic CSV I/O helpers used by the reproducibility scripts."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np


def results_dir() -> Path:
    p = Path(os.environ.get("GSL_RESULTS_DIR", "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def result_path(name: str) -> Path:
    return results_dir() / name


def save_csv(name: str, header, rows, fmt="%.16g") -> Path:
    """Write a numeric CSV atomically and return its path."""
    path = result_path(name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr = np.asarray(list(rows), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    np.savetxt(tmp, arr, delimiter=",", header=",".join(header), comments="", fmt=fmt)
    tmp.replace(path)
    print(f"wrote {path}")
    return path

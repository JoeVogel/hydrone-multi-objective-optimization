# -*- coding: utf-8 -*-
"""
Overlay |Cp|min vs alpha with CL vs alpha on a secondary axis.

Inputs:
  --cpmin_csv   CSV with columns: angle_deg, abs_cp_min (and optionally file, cp_min, n_points)
  --polar_csv   XFOIL polar CSV (may include header prose before the 'Alpha,Cl,...' row)
  --out_png     Output PNG path (optional; if omitted, just shows the figure)
  --title       Custom plot title (optional)
  --xlim        X-axis limits, e.g. --xlim -5 20
  --alpha_max   Hard cap for alpha used in CLmax search (default: 20 deg)
  --cl_abs_max  Discard |CL| above this value as non-physical (default: 1.8)
  --cd_max      Discard Cd above this value (default: 0.4)

Usage
-----
python overlay_cpmin_polar.py \
  --cpmin_csv cpmin_e63_10000.csv \
  --polar_csv E63_10000.csv \
  --out_png overlay_cpmin_polar_e63_10000.png
  --xlim -5 20

Optional:
  --out_csv merged_cpmin_polar_e63_10000.csv
  --foil_name E63
  --title "E63 - Re=10000, Mach=0"

Assumptions
-----------
- cpmin_csv must contain: angle_deg, abs_cp_min
- polar_csv must contain: alpha (or ALFA) and CL (or Cl/c_l)
- All angles are in degrees.

Output
------
- PNG figure with |Cp|min (left y-axis) and CL (right y-axis)
- Marks α where |Cp|min reaches its maximum and α where CL reaches CL_max
- Optional merged CSV with alpha_deg, abs_cp_min, CL
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_cpmin_csv(path: Path) -> pd.DataFrame:
    """Read |Cp|min CSV and return DataFrame with columns alpha_deg, cpmin_abs."""
    df = pd.read_csv(path)
    # Accept flexible column names
    col_alpha = None
    for cand in ["angle_deg", "alpha_deg", "alpha", "Alpha"]:
        if cand in df.columns:
            col_alpha = cand
            break
    if col_alpha is None:
        raise ValueError("Could not find angle column in cpmin CSV (expected 'angle_deg').")

    col_cpabs = None
    for cand in ["abs_cp_min", "cpmin_abs", "|cp|min", "cp_abs"]:
        if cand in df.columns:
            col_cpabs = cand
            break
    if col_cpabs is None:
        # Fall back to compute abs if raw Cp provided
        if "cp_min" in df.columns:
            df["abs_cp_min"] = df["cp_min"].abs()
            col_cpabs = "abs_cp_min"
        else:
            raise ValueError("Could not find |Cp|min column in cpmin CSV (e.g., 'abs_cp_min').")

    out = df[[col_alpha, col_cpabs]].copy()
    out.columns = ["alpha_deg", "cpmin_abs"]
    out = out.dropna().sort_values("alpha_deg").reset_index(drop=True)
    return out


def _find_header_line(lines) -> int:
    """Find the index of the header row that starts with 'Alpha,Cl'."""
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("alpha,cl"):
            return i
    # Some polars might use semicolon or whitespace; try a loose match
    for i, ln in enumerate(lines):
        tokens = [t.strip().lower() for t in ln.replace(";", ",").split(",")]
        if len(tokens) >= 2 and tokens[0] == "alpha" and tokens[1] in ("cl", "c_l"):
            return i
    raise ValueError("Could not find 'Alpha,Cl,...' header line in polar CSV.")


def read_polar_csv(path: Path) -> pd.DataFrame:
    """Read XFOIL polar with a possible verbose preamble and return DataFrame with standard names."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    hdr_idx = _find_header_line(lines)
    # Read from header line onwards
    dat = pd.read_csv(
        Path(path),
        skiprows=hdr_idx,
        sep=r"[,\s]+",
        engine="python"
    )
    # Standardize column names
    rename = {}
    for c in dat.columns:
        lc = str(c).strip().lower()
        if lc == "alpha":
            rename[c] = "alpha_deg"
        elif lc in ("cl", "c_l"):
            rename[c] = "CL"
        elif lc in ("cd", "c_d"):
            rename[c] = "Cd"
        elif lc in ("cdp", "c_dp"):
            rename[c] = "Cdp"
        elif lc in ("cm", "c_m"):
            rename[c] = "Cm"
        elif "top" in lc and "xtr" in lc:
            rename[c] = "Top_Xtr"
        elif "bot" in lc and "xtr" in lc:
            rename[c] = "Bot_Xtr"
    dat = dat.rename(columns=rename)

    # Keep only essentials
    keep = [c for c in ["alpha_deg", "CL", "Cd", "Cdp", "Cm", "Top_Xtr", "Bot_Xtr"] if c in dat.columns]
    dat = dat[keep].copy()

    # Basic cleaning
    dat = dat.replace([np.inf, -np.inf], np.nan).dropna(subset=["alpha_deg", "CL"])
    dat = dat.sort_values("alpha_deg").reset_index(drop=True)
    return dat


def clean_polar(df: pd.DataFrame, cl_abs_max: float, cd_max: float, alpha_max: float) -> pd.DataFrame:
    """Filter non-physical points and limit the range to alpha <= alpha_max."""
    out = df.copy()

    # Drop absurd CL
    out = out[np.abs(out["CL"]) <= cl_abs_max]

    # Drop huge Cd if present (often indicates separated, non-converged solutions)
    if "Cd" in out.columns:
        out = out[out["Cd"] <= cd_max]

    # Keep alpha within a reasonable cap for Re~low
    out = out[out["alpha_deg"] <= alpha_max]

    # Remove duplicates and ensure monotonic alpha
    out = out.drop_duplicates(subset=["alpha_deg"], keep="first")
    out = out.sort_values("alpha_deg").reset_index(drop=True)
    return out


def detect_clmax_valid(df: pd.DataFrame, drop_ratio: float = 0.05, lookahead: int = 5) -> tuple[float, float]:
    """
    Detect a physically-meaningful CLmax:
      - Find the first local maximum where CL drops by at least 'drop_ratio' within 'lookahead' points ahead.
      - If none found, fallback to global max in the provided (cleaned) range.

    Returns (alpha_at_clmax, clmax).
    """
    alpha = df["alpha_deg"].to_numpy()
    cl = df["CL"].to_numpy()
    n = len(df)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(alpha[0]), float(cl[0])

    # First-local-maximum criterion
    for i in range(1, n - 1):
        if cl[i] >= cl[i - 1] and cl[i] >= cl[i + 1]:
            j_end = min(n, i + 1 + lookahead)
            future = cl[i + 1:j_end]
            if future.size > 0 and np.nanmin(future) <= cl[i] * (1.0 - drop_ratio):
                return float(alpha[i]), float(cl[i])

    # Fallback: global max
    imax = int(np.nanargmax(cl))
    return float(alpha[imax]), float(cl[imax])


def main():
    ap = argparse.ArgumentParser(description="Overlay |Cp|min and CL with robust CLmax detection.")
    ap.add_argument("--cpmin_csv", type=Path, required=True, help="CSV with angle_deg & abs_cp_min.")
    ap.add_argument("--polar_csv", type=Path, required=True, help="XFOIL polar CSV.")
    ap.add_argument("--out_png", type=Path, default=None, help="Save figure to this PNG.")
    ap.add_argument("--title", type=str, default=None, help="Custom plot title.")
    ap.add_argument("--xlim", nargs=2, type=float, default=[-5.0, 20.0], help="X-axis limits: min max.")
    ap.add_argument("--alpha_max", type=float, default=20.0, help="Max alpha to consider for CLmax search.")
    ap.add_argument("--cl_abs_max", type=float, default=1.8, help="Discard |CL| above this as non-physical.")
    ap.add_argument("--cd_max", type=float, default=0.4, help="Discard Cd above this as non-physical.")
    args = ap.parse_args()

    # Read inputs
    df_cp = read_cpmin_csv(args.cpmin_csv)
    df_pol_raw = read_polar_csv(args.polar_csv)
    df_pol = clean_polar(df_pol_raw, cl_abs_max=args.cl_abs_max, cd_max=args.cd_max, alpha_max=args.alpha_max)

    # Maxima
    i_cp = int(np.nanargmax(df_cp["cpmin_abs"].to_numpy()))
    alpha_cpmax = float(df_cp.loc[i_cp, "alpha_deg"])
    cpmax = float(df_cp.loc[i_cp, "cpmin_abs"])

    alpha_clmax, clmax = detect_clmax_valid(df_pol, drop_ratio=0.05, lookahead=5)

    # Title
    if args.title:
        title = args.title
    else:
        # Try to infer RE and Mach from polar prose (optional)
        title = "Overlay |Cp|min vs CL"
        # If your filenames encode Re/Mach/airfoil, adapt here:
        # title = f"{airfoil} - Re={Re}, Mach={Mach}"

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    # Curves (explicitly set different colors per your request)
    l1, = ax.plot(df_cp["alpha_deg"], df_cp["cpmin_abs"], label="|Cp|min", color="tab:red", lw=2.0)
    l2, = ax2.plot(df_pol["alpha_deg"], df_pol["CL"], label="CL", color="tab:blue", lw=2.0)

    # Annotations
    ax.axvline(alpha_cpmax, color="tab:red", ls="--", lw=1.2, alpha=0.6)
    ax2.axvline(alpha_clmax, color="tab:blue", ls="--", lw=1.2, alpha=0.6)

    ax.annotate(
        f"α@|Cp|min_max = {alpha_cpmax:.2f}°",
        xy=(alpha_cpmax, cpmax),
        xytext=(alpha_cpmax + 0.5, cpmax + 0.15),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        color="tab:red",
        fontsize=10,
    )
    ax2.annotate(
        f"α@CLmax = {alpha_clmax:.2f}°",
        xy=(alpha_clmax, clmax),
        xytext=(alpha_clmax + 0.5, clmax + 0.05),
        arrowprops=dict(arrowstyle="->", color="tab:blue"),
        color="tab:blue",
        fontsize=10,
    )

    # Labels, limits, grid
    ax.set_xlabel(r"$\alpha$ (deg)")
    ax.set_ylabel(r"$|C_p|_{\min}$")
    ax2.set_ylabel("CL")
    ax.set_xlim(args.xlim[0], args.xlim[1])
    ax.grid(True, ls=":", alpha=0.5)

    # Legend
    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="upper left")

    # Title last (so it doesn't get overridden)
    ax.set_title(title, pad=12)

    fig.tight_layout()

    if args.out_png:
        out = Path(args.out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

"""
Module for holding airfoil data, and providing drag and lift coefficients to the solver.

Airfoil data is stored in the folder airfoils. Currently, the Aerodyn format and AirfoilTools CSV format
are supported, with only a single airfoil table. Data must be available from -180 to 180 degrees,
and a quadratic function is built to interpolate between data in the airfoil table.

This module can also be executed from the command-line to plot drag and lift coefficients 
for a single airfoil, e.g. 

.. code-block:: console

    python airfoil.py NACA_4412

    or

    python airfoil.py NACA_4412 -180 180

    or 

    python airfoil.py NACA_4412 -180 180 10000

    or 

    python airfoil.py NACA_4412 -180 180 10000 cpmin

"""

import os
import sys
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import degrees, radians, atan2, sin, cos
from bisect import bisect_left
from typing import Optional, Union, Iterable


AIRFOILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'airfoils')
CPMIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpmin')

def _read_airfoiltools_header(path):
    """
    Read the header (before line 'Alpha,...') of CSV from AirfoilTools
    and return meta information and the line index of the table start:
      - meta: dict with key->value pairs (e.g.: {'Reynolds number': '100000', 'Ncrit': '9', ...})
      - start_idx: 0-based index of the table header (which starts with  'Alpha')
    """
    meta = {}
    start_idx = None
    with open(path, 'r', encoding='utf-8') as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue

            if line.startswith('Alpha'):
                start_idx = i
                break
            
            if ',' in line:
                key, val = line.split(',', 1)
                meta[key.strip()] = val.strip()
    if start_idx is None:
        raise ValueError(f"Alpha not found in: {path}")
    return meta, start_idx

def _read_polar_file_with_meta(path):
    """
    Reads a polar file (.csv AirfoilTools or .dat XFOIL).
    Returns: (alpha_deg, Cl, Cd, Re, stall_alpha or None)
    """
    import os
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        meta, start = _read_airfoiltools_header(path)
        
        df = pd.read_csv(path, skiprows=start)
        alpha = df['Alpha'].to_numpy(dtype=float)
        Cl    = df['Cl'].to_numpy(dtype=float)
        Cd    = df['Cd'].to_numpy(dtype=float)

        # Reynolds from header
        Re = None
        if 'Reynolds number' in meta:
            try:
                Re = float(meta['Reynolds number'])
            except ValueError:
                raise ValueError(f"Invalid Reynolds number in header of: {path}")
        else:
            raise ValueError(f"Reynolds number not found in header : {path}")
        
        # Stall alpha
        stall_alpha = None
        if 'Stall alpha' in meta:
            try:
                stall_alpha = float(meta['Stall alpha'])
            except ValueError:
                stall_alpha = None

        return alpha, Cl, Cd, Re, stall_alpha

    elif ext == '.dat':
        # XFOIL .dat "clássico": não tem meta com Re
        alpha, Cl, Cd = np.loadtxt(path, skiprows=14, unpack=True)
        return alpha, Cl, Cd, None, None

    else:
        raise ValueError(f"File format not supported: {path}")

def _read_cpmin_files(name: str):
    """
    Load cp_min CSV files for an airfoil from ./cpmin.
    Expected columns: angle_deg, cp_min. Cp_min is negative.
    Filenames: cpmin_<airfoil>_<Re>.csv  (case-insensitive, e.g., cpmin_naca4412_100000.csv)

    Cpmin refers to the minimum value of the pressure coefficient (\(C_{p}\)) on the airfoil's surface at a 
    given angle of attack. This value indicates the point of lowest pressure, which is a critical parameter for 
    understanding flow behavior, as it can precede flow separation and is directly related to the airfoil's
    lift characteristics

    Returns a list of (Re, alpha_deg_array, cpmin_array).
    """
    norm = re.sub(r'[^a-z0-9]', '', name.lower())
    files = glob.glob(os.path.join(CPMIN_DIR, f'cpmin_{norm}_*.csv'))
    out = []
    for path in files:
        m = re.search(rf'cpmin_{norm}_(\d+)\.csv$', os.path.basename(path), flags=re.IGNORECASE)
        if not m:
            continue
        Re = float(m.group(1))
        df = pd.read_csv(path)
        alpha = df['angle_deg'].to_numpy(dtype=float)
        cpmin = df['cp_min'].to_numpy(dtype=float)
        out.append((Re, alpha, cpmin))
    out.sort(key=lambda t: t[0])
    return out

def thickness_e63() -> float:
    """
    Relative thickness t/c for the Eppler 63 airfoil based on airfoil tools.
    http://airfoiltools.com/airfoil/details?airfoil=e63-il
    """
    
    return float(0.0425) # at 22.8% chord.

def thickness_naca(foil_name: str) -> float:
        """
        Estimate relative thickness (t/c) from the foil name.

        Rules:
        - NACA 4- or 5-digit: e.g. NACA0018, NACA23012 → last two digits = thickness (%)
        - NACA 6-series: e.g. NACA63-018, NACA66-209 → digits after dash = thickness (%)
        - Returns 0.12 as default if unknown.
        """
        if not isinstance(foil_name, str):
            return 0.12
        foil = foil_name.upper().replace(" ", "")
        m = re.match(r"^NACA(\d{4,5})$", foil)
        if m:
            digits = m.group(1)
            return int(digits[-2:]) / 100.0
        m = re.match(r"^NACA\d{2}-?(\d{2,3})$", foil)
        if m:
            return int(m.group(1)) / 100.0
        return 0.12

class Airfoil:
    """
    Class for storing airfoil drag and lift coefficients. Should be initialized using 
    the load_airfoil() function.
    """

    def __init__(self):
        self.name = None

        # single-Re (backward compatibility)
        self.alpha_ = None
        self.Cl_ = None
        self.Cd_ = None
        self.Cl_func = None
        self.Cd_func = None

        # Multi-Re storage
        self.re_list = []       # sorted list of Re (floats)
        self._cl_funcs = {}     # Re -> interp1d(alpha_deg -> Cl)
        self._cd_funcs = {}     # Re -> interp1d(alpha_deg -> Cd)
        self.default_Re = None  # chosen default Re when caller omits Re

        # Cp_min (multi-Re or single-Re)
        self._cp_min_funcs = {}   # Re -> interp1d(alpha_deg -> cp_min)
        self._cpmin_re_list = []  # sorted Re list for cp_min
        self.Cp_min_func = None   # single-Re fallback
        self.cpmin_alpha_ = None  # single-Re alpha cache
        
        # Stall alpha (multi-Re)
        self._stall_alpha = {}      # Re -> stall_alpha_deg (float)
        self._stall_re_list = []    # sorted list of Re for which stall is available

        self.zero_lift = 0.0
        
        # --- Out-of-range alpha handling (validation vs optimization) ---
        #  - "nan": return NaN outside trusted range (recommended for optimization)
        #  - "clamp": clamp alpha to trusted range
        #  - "clamp_drag_penalty": clamp Cl/Cpmin, but increase Cd outside range (recommended for validation)
        self.out_of_range_policy = "nan"
        self.drag_penalty_k = 0.02  # Cd *= (1 + k * delta_alpha_deg^2) when outside trusted range. May need some fine tune.
        self.enable_oor_logging = False
        
    def _normalize_angle(self, alpha):
        """
        Ensure that the angle fulfils :math:`\\pi < \\alpha < \\pi`

        :param float alpha: Angle in radians
        :return: Normalized angle
        :rtype: float
        """

        return atan2(sin(alpha), cos(alpha))
    
    def _sanitize_re(self, Re: Optional[float]) -> float:
        """Ensure Reynolds is finite and strictly positive for log(Re) interpolation."""
        if not self.re_list:
            return float(Re) if Re is not None else float("nan")

        if Re is None:
            return float(self.default_Re or self.re_list[len(self.re_list)//2])

        Re = float(Re)
        if (not np.isfinite(Re)) or (Re <= 0.0):
            return float(self.default_Re or self.re_list[0])

        # clamp to minimum available to avoid extreme low-Re extrapolation
        Re_min = float(self.re_list[0])
        if Re < Re_min:
            return Re_min
        return Re

    def _eval_clipped(self, func: interp1d, alpha_deg: float) -> float:
        """Evaluate interp1d safely by clamping alpha to its support."""
        a_min = float(np.min(func.x))
        a_max = float(np.max(func.x))
        a = max(a_min, min(a_max, float(alpha_deg)))
        return float(func(a))

    def _trusted_alpha_limit_deg(self, Re: Optional[float] = None):
        """Return (limit_deg, data_abs_max_deg) for alpha lookup."""
        if self.re_list:
            use_Re = self._sanitize_re(Re)
            func = self._cl_funcs.get(use_Re, self._cl_funcs[self.re_list[0]])
            data_abs_max = float(np.max(np.abs(func.x)))
        else:
            data_abs_max = float(np.max(np.abs(self.alpha_))) if self.alpha_ is not None else float("nan")

        stall = float("nan")
        try:
            stall = float(self.stall_alpha(Re))
        except Exception:
            stall = float("nan")

        limit = min(abs(stall), data_abs_max) if np.isfinite(stall) else data_abs_max
        return float(limit), float(data_abs_max)

    def _apply_oor_policy(self, alpha_deg: float, Re: Optional[float] = None):
        """Apply out-of-range alpha policy. Returns (alpha_eff_deg, delta_deg, limit_deg)."""
        limit_deg, _ = self._trusted_alpha_limit_deg(Re)
        if not np.isfinite(limit_deg):
            return float("nan"), float("nan"), float("nan")

        abs_a = abs(float(alpha_deg))
        delta = max(0.0, abs_a - limit_deg)

        if delta <= 0.0:
            return float(alpha_deg), 0.0, float(limit_deg)

        pol = getattr(self, "out_of_range_policy", "nan")
        if pol == "nan":
            return float("nan"), float(delta), float(limit_deg)

        alpha_clamped = float(np.sign(alpha_deg) * limit_deg)
        if getattr(self, "enable_oor_logging", False):
            print(f"[Airfoil:{self.name}] alpha out-of-range: {alpha_deg:.3f} deg (limit {limit_deg:.3f}), policy={pol}")
        return alpha_clamped, float(delta), float(limit_deg)

    
    # ---------- Public API ----------

    def Cl(self, alpha, Re=None):
        """
        Lift coefficient at given angle of attack (in radians) and Reynolds number.
        If multi-Re and Re is None, uses the default_Re (median of available Re).
        If only single-Re data is available, Re is ignored.
        Out-of-range alpha handling is controlled by self.out_of_range_policy
        :param float alpha: Angle of attack in radians
        :param float Re: Reynolds number (optional)
        """
        alpha_deg = -self.zero_lift + degrees(self._normalize_angle(alpha))
        
        alpha_eff, _, _ = self._apply_oor_policy(alpha_deg, Re)
        if np.isnan(alpha_eff):
            return float("nan")
            
        if self.re_list:
            use_Re = self._sanitize_re(Re)
            return self._cl_alpha_re(alpha_eff, use_Re)

        return float(self.Cl_func(alpha_eff))

    def Cd(self, alpha, Re=None):
        """
        Drag coefficient at given angle of attack (in radians) and Reynolds number.
        Out-of-range alpha handling is controlled by self.out_of_range_policy.
        If policy is 'clamp_drag_penalty', Cd is increased outside trusted range.
        :param float alpha: Angle of attack in radians
        :param float Re: Reynolds number (optional)
        """
        alpha_deg = -self.zero_lift + degrees(self._normalize_angle(alpha))
        
        alpha_eff, delta_deg, _ = self._apply_oor_policy(alpha_deg, Re)
        if np.isnan(alpha_eff):
            return float("nan")
        
        if self.re_list:
            use_Re = self._sanitize_re(Re)
            cd = self._cd_alpha_re(alpha_eff, use_Re)
        else:
            cd = float(self.Cd_func(alpha_eff))
            
        pol = getattr(self, "out_of_range_policy", "nan")
        if pol == "clamp_drag_penalty" and delta_deg > 0.0 and np.isfinite(cd):
            k = float(getattr(self, "drag_penalty_k", 0.02))
            cd = cd * (1.0 + k * (delta_deg ** 2))

        return float(cd)

    def Cp_min(self, alpha, Re=None):
        """
        Minimum pressure coefficient at given angle (radians) and Reynolds number.
        Uses bilinear interpolation in (alpha, Re). If only one Re is available, the Re argument is ignored.
        Out-of-range alpha handling is controlled by self.out_of_range_policy.
        :param float alpha: Angle of attack in radians
        :param float Re: Reynolds number (optional)
        """
        alpha_deg = -self.zero_lift + degrees(self._normalize_angle(alpha))
        
        alpha_eff, _, _ = self._apply_oor_policy(alpha_deg, Re)
        if np.isnan(alpha_eff):
            return float("nan")
        
        if self._cpmin_re_list:
            use_Re = Re if Re is not None else (self._cpmin_re_list[len(self._cpmin_re_list)//2])
            use_Re = float(use_Re)
            if (not np.isfinite(use_Re)) or (use_Re <= 0.0):
                use_Re = float(self._cpmin_re_list[0])
            return self._cpmin_bilinear(alpha_eff, use_Re)

        if self.Cp_min_func is None:
            raise ValueError(f'cp_min data not loaded for {self.name}')
            
        return float(self.Cp_min_func(alpha_eff))

    def stall_alpha(self, Re=None):
        """
        Returns stall alpha in degrees for a given Reynolds number.
        If multi-Re and Re is None, uses default_Re (or median).
        If stall data is missing, returns NaN.
        """
        if not self._stall_re_list:
            return float("nan")

        use_Re = Re if Re is not None else (self.default_Re or self._stall_re_list[len(self._stall_re_list)//2])
        try:
            use_Re = float(use_Re)
        except Exception:
            use_Re = float(self.default_Re or self._stall_re_list[0])

        if (not np.isfinite(use_Re)) or (use_Re <= 0.0):
            use_Re = float(self.default_Re or self._stall_re_list[0])

        # exact match
        if use_Re in self._stall_alpha:
            return float(self._stall_alpha[use_Re])

        # interpolate
        re_sorted = self._stall_re_list
        if len(re_sorted) == 1:
            return float(self._stall_alpha[re_sorted[0]])

        i = bisect_left(re_sorted, use_Re)

        if i == 0:
            r1, r2 = re_sorted[0], re_sorted[1]
        elif i >= len(re_sorted):
            r1, r2 = re_sorted[-2], re_sorted[-1]
        else:
            r1, r2 = re_sorted[i-1], re_sorted[i]

        v1 = float(self._stall_alpha[r1])
        v2 = float(self._stall_alpha[r2])

        w = (np.log(use_Re) - np.log(r1)) / (np.log(r2) - np.log(r1))
        return float((1.0 - w) * v1 + w * v2)


    # ---------- Internal helpers for multi-Re ----------
    def _cl_alpha_re(self, alpha_deg, Re):
        if Re in self._cl_funcs:
            return self._eval_clipped(self._cl_funcs[Re], alpha_deg)
        return self._interp_in_logRe(alpha_deg, Re, self._cl_funcs)

    def _cd_alpha_re(self, alpha_deg, Re):
        if Re in self._cd_funcs:
            return self._eval_clipped(self._cd_funcs[Re], alpha_deg)
        return self._interp_in_logRe(alpha_deg, Re, self._cd_funcs)

    def _interp_in_logRe(self, alpha_deg, Re, func_dict):
        # find neighbors in re_list
        re_sorted = self.re_list
        
        try:
            Re = float(Re)
        except Exception:
            Re = float(re_sorted[0])
            
        if (not np.isfinite(Re)) or (Re <= 0.0):
            Re = float(re_sorted[0])
        
        if len(re_sorted) == 1:
            return self._eval_clipped(func_dict[re_sorted[0]], alpha_deg)
            
        logRe = np.log(Re)
        logRe_list = np.log(np.array(re_sorted))
        i = bisect_left(re_sorted, Re)

        if i == 0:
            # below smallest: extrapolate linearly in logRe
            f1 = self._eval_clipped(func_dict[re_sorted[0]], alpha_deg)
            f2 = self._eval_clipped(func_dict[re_sorted[1]], alpha_deg)
            w = (logRe - logRe_list[0]) / (logRe_list[1] - logRe_list[0])
            return float((1 - w) * f1 + w * f2)

        if i >= len(re_sorted):
            # above largest: extrapolate
            f1 = self._eval_clipped(func_dict[re_sorted[-2]], alpha_deg)
            f2 = self._eval_clipped(func_dict[re_sorted[-1]], alpha_deg)
            w = (logRe - logRe_list[-2]) / (logRe_list[-1] - logRe_list[-2])
            return float((1 - w) * f1 + w * f2)

        # between i-1 and i
        re1, re2 = re_sorted[i-1], re_sorted[i]
        f1 = self._eval_clipped(func_dict[re1], alpha_deg)
        f2 = self._eval_clipped(func_dict[re2], alpha_deg)
        w = (logRe - np.log(re1)) / (np.log(re2) - np.log(re1))
        return float((1 - w) * f1 + w * f2)

    def _cpmin_bilinear(self, alpha_deg: float, Re: float) -> float:
        """
        Bilinear interpolation for cp_min:
        - Linear interpolation in angle within each Re.
        - Linear interpolation in Re (not log(Re)).
        """
        re_sorted = self._cpmin_re_list
        if not re_sorted:
            raise ValueError(f'cp_min data not loaded for {self.name}')
            
        try:
            Re = float(Re)
        except Exception:
            Re = float(re_sorted[0])

        if (not np.isfinite(Re)) or (Re <= 0.0):
            Re = float(re_sorted[0])

        # Exact match
        if Re in self._cp_min_funcs:
            return self._eval_clipped(self._cp_min_funcs[Re], alpha_deg)

        if len(re_sorted) == 1:
            return self._eval_clipped(self._cp_min_funcs[re_sorted[0]], alpha_deg)

        # Find bounding Re values
        i = bisect_left(re_sorted, Re)
        if i == 0:
            r1, r2 = re_sorted[0], re_sorted[1]
        elif i >= len(re_sorted):
            r1, r2 = re_sorted[-2], re_sorted[-1]
        else:
            r1, r2 = re_sorted[i - 1], re_sorted[i]

        v1 = self._eval_clipped(self._cp_min_funcs[r1], alpha_deg)
        v2 = self._eval_clipped(self._cp_min_funcs[r2], alpha_deg)

        # Linear interpolation in Re
        w = (Re - r1) / (r2 - r1)
        return float((1.0 - w) * v1 + w * v2)

    # ---------- Plot (single-Re or any chosen Re) ----------

    def plot(self, color='k', Re=None, xlim=None):
        """
        If multi-Re and Re provided: plot Cl/Cd vs alpha sampling only
        the valid alpha range for the involved polar(s).
        If xlim=(xmin, xmax) is provided, also restricts to that range.
        """

        # Single-Re case (old behavior)
        if not self.re_list:
            plt.plot(self.alpha_, self.Cl_, color + '-')
            plt.plot(self.alpha_, self.Cd_, color + '--')
            plt.title(f'Airfoil characteristics for {self.name}')
            plt.xlabel('Angle of attack (deg)')
            plt.ylabel('Drag and lift coefficients')
            plt.legend(('$C_l$','$C_d$'))
            return

        # Multi-Re case
        if Re is None:
            Re = self.re_list[len(self.re_list)//2]

        re_sorted = self.re_list

        # 1) Define the allowed alpha range
        if Re in self._cl_funcs:
            # Exact Re: use the alpha grid of this polar
            a_min = self._cl_funcs[Re].x.min()
            a_max = self._cl_funcs[Re].x.max()
        else:
            # Re between neighbors: use the INTERSECTION of their alpha ranges
            i = bisect_left(re_sorted, Re)
            i = max(1, min(i, len(re_sorted)-1))
            Re1, Re2 = re_sorted[i-1], re_sorted[i]
            a1 = self._cl_funcs[Re1].x
            a2 = self._cl_funcs[Re2].x
            a_min = max(a1.min(), a2.min())
            a_max = min(a1.max(), a2.max())
            # If intersection is empty (or almost empty), fall back to the closest neighbor
            if a_max <= a_min + 1e-12:
                use_Re = Re1 if abs(np.log(Re) - np.log(Re1)) < abs(np.log(Re) - np.log(Re2)) else Re2
                a_min = self._cl_funcs[use_Re].x.min()
                a_max = self._cl_funcs[use_Re].x.max()

        # 2) Apply xlim coming from CLI (optional)
        if xlim is not None:
            lo, hi = xlim
            a_min = max(a_min, lo)
            a_max = min(a_max, hi)

        # Numerical safeguard
        if a_max <= a_min + 1e-12:
            alpha_grid = np.array([a_min, a_max])
        else:
            alpha_grid = np.linspace(a_min, a_max, 400)

        # 3) Evaluate Cl, Cd on the chosen grid
        Cl_vals = [self._cl_alpha_re(a, Re) for a in alpha_grid]
        Cd_vals = [self._cd_alpha_re(a, Re) for a in alpha_grid]

        # 4) Plot curves
        plt.plot(alpha_grid, Cl_vals, color + '-')
        plt.plot(alpha_grid, Cd_vals, color + '--')
        plt.title(f'Airfoil characteristics for {self.name} @ Re={Re:.0f}')
        plt.xlabel('Angle of attack (deg)')
        plt.xticks(np.arange(a_min, a_max + 1, 2))
        plt.ylabel('Drag and lift coefficients')
        plt.legend(('$C_l$', '$C_d$'))
        plt.grid(True, linestyle='--', alpha=0.6)
    
    def plot_cpmin(self, Re=None, xlim=None):
        """
        Plot cp_min vs angle (degrees) for a chosen Re.
        """
        if not self._cpmin_re_list and self.Cp_min_func is None:
            raise ValueError(f'cp_min data not loaded for {self.name}')

        # choose Re and alpha grid
        if self._cpmin_re_list:
            if Re is None:
                Re = self._cpmin_re_list[len(self._cpmin_re_list)//2]
            if Re in self._cp_min_funcs:
                a_min = float(self._cp_min_funcs[Re].x.min())
                a_max = float(self._cp_min_funcs[Re].x.max())
            else:
                i = bisect_left(self._cpmin_re_list, Re)
                i = max(1, min(i, len(self._cpmin_re_list)-1))
                r1, r2 = self._cpmin_re_list[i-1], self._cpmin_re_list[i]
                a1 = self._cp_min_funcs[r1].x
                a2 = self._cp_min_funcs[r2].x
                a_min = max(a1.min(), a2.min())
                a_max = min(a1.max(), a2.max())
        else:
            # single-Re
            a_min = float(self.cpmin_alpha_.min())
            a_max = float(self.cpmin_alpha_.max())

        if xlim is not None:
            a_min = max(a_min, float(xlim[0]))
            a_max = min(a_max, float(xlim[1]))

        alpha_grid = np.linspace(a_min, a_max, 400) if a_max > a_min else np.array([a_min, a_max])

        if self._cpmin_re_list:
            y = [self._cpmin_bilinear(a, Re) for a in alpha_grid]
            title_re = f' @ Re={Re:.0f}'
        else:
            y = self.Cp_min_func(alpha_grid)
            title_re = ''

        plt.plot(alpha_grid, y, '-')
        plt.title(f'cp_min for {self.name}{title_re}')
        plt.xlabel('Angle of attack (deg)')
        plt.ylabel('cp_min (negative)')
        plt.xticks(np.arange(a_min, a_max + 1, 2))
        plt.grid(True, linestyle='--', alpha=0.6)
        
def load_airfoil(name):
    """
    Backward-compatible loader:
      - If only one polar is found, builds single-Re Airfoil as before.
      - If multiple polars with Re in file names are found, builds multi-Re structure.
    Supported:
      - {name}.dat    (classic XFOIL format)  [single-Re unless you duplicate with Re in name]
      - {name}_*.csv  (AirfoilTools CSV)
      - {name}_*.dat  (if you export .dat with Re in name)
    """
    a = Airfoil()
    a.name = name

    base_dir = AIRFOILS_DIR
    
    candidates = []
    candidates.extend(glob.glob(os.path.join(base_dir, f'{name}.dat')))
    candidates.extend(glob.glob(os.path.join(base_dir, f'{name}_*.dat')))
    candidates.extend(glob.glob(os.path.join(base_dir, f'{name}_*.csv')))
    candidates.extend(glob.glob(os.path.join(base_dir, f'{name}.csv')))  # se existir

    if not candidates:
        raise FileNotFoundError(f'No polar files for airfoil {name}')

    # Carrega todos e separa por Re
    with_re = []   # lista de (Re, alpha, Cl, Cd, stall_alpha)
    defaults = []  # sem Re detectado

    for p in candidates:
        alpha, Cl, Cd, Re, stall_alpha = _read_polar_file_with_meta(p)
        if Re is not None:
            with_re.append((Re, alpha, Cl, Cd, stall_alpha))
        else:
            defaults.append((alpha, Cl, Cd))

    # --- Load cp_min tables (if present) ---
    cpmin_tables = _read_cpmin_files(name)
    if cpmin_tables:
        a._cpmin_re_list = [t[0] for t in cpmin_tables]
        if len(cpmin_tables) == 1:
            # single-Re fallback
            _, alpha_deg_c, cpmin_c = cpmin_tables[0]
            a.cpmin_alpha_ = alpha_deg_c
            a.Cp_min_func = interp1d(alpha_deg_c, cpmin_c, kind='linear',
                                     bounds_error=False, fill_value=(cpmin_c[0], cpmin_c[-1]))
        else:
            for Re_val, alpha_deg_c, cpmin_c in cpmin_tables:
                a._cp_min_funcs[Re_val] = interp1d(alpha_deg_c, cpmin_c, kind='linear',
                                     bounds_error=False, fill_value=(cpmin_c[0], cpmin_c[-1]))

    if with_re:
        # Multi-Re: ordena por Re
        with_re.sort(key=lambda t: t[0])
        a.re_list = [t[0] for t in with_re]
        for Re_val, alpha_deg, Cl, Cd, stall_a in with_re:
            clf = interp1d(alpha_deg, Cl, kind='linear', bounds_error=False, fill_value=(Cl[0], Cl[-1]))
            cdf = interp1d(alpha_deg, Cd, kind='linear', bounds_error=False, fill_value=(Cd[0], Cd[-1]))
            a._cl_funcs[Re_val] = clf
            a._cd_funcs[Re_val] = cdf
            
            if stall_a is not None and not np.isnan(stall_a):
                a._stall_alpha[Re_val] = float(stall_a)
        
        a._stall_re_list = sorted(a._stall_alpha.keys())
        
        a.default_Re = a.re_list[len(a.re_list)//2]  # pick median Re as default
        
        return a

    # Fallback: single-Re (nenhum arquivo tinha 'Reynolds number' no cabeçalho)
    if defaults:
        alpha_deg, Cl, Cd = defaults[0]
        a.alpha_, a.Cl_, a.Cd_ = alpha_deg, Cl, Cd
        a.Cl_func = interp1d(a.alpha_, a.Cl_, kind='linear', bounds_error=False, fill_value=(a.Cl_[0], a.Cl_[-1]))
        a.Cd_func = interp1d(a.alpha_, a.Cd_, kind='linear', bounds_error=False, fill_value=(a.Cd_[0], a.Cd_[-1]))
        return a

    raise FileNotFoundError(f'No usable polar files for airfoil {name}')


if __name__ == '__main__':
    # Load and plot airfoil from command line
    name = sys.argv[1]
    # name = 'NACA_0018'
    a = load_airfoil(name)
    x_lower_limit = -180
    x_upper_limit = 180
    Re_cli = None

    # Usage:
    #   python airfoil.py NAME
    #   python airfoil.py NAME <xmin> <xmax>
    #   python airfoil.py NAME <xmin> <xmax> <Re> 
    #   python airfoil.py NAME <xmin> <xmax> <Re> cpmin
    if len(sys.argv) > 3:
        x_lower_limit = float(sys.argv[2])
        x_upper_limit = float(sys.argv[3])
    if len(sys.argv) > 4:
        Re_cli = float(sys.argv[4])

    ax = plt.gca()
    ax.set_xlim(x_lower_limit, x_upper_limit)
    
    if len(sys.argv) > 5 and sys.argv[5].lower() == 'cpmin':
        a.plot_cpmin(Re=Re_cli, xlim=(x_lower_limit, x_upper_limit))
    else:
        a.plot(Re=Re_cli)

    plt.show()

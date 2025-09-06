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

    python airfoil.py NACA_4412 -180 180 100000

"""

import os
import sys
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from math import degrees, radians, atan2, sin, cos
from bisect import bisect_left


AIRFOILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'airfoils')

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
    Returns: (alpha_deg, Cl, Cd, Re or None)
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

        return alpha, Cl, Cd, Re

    elif ext == '.dat':
        # XFOIL .dat "clássico": não tem meta com Re
        alpha, Cl, Cd = np.loadtxt(path, skiprows=14, unpack=True)
        return alpha, Cl, Cd, None

    else:
        raise ValueError(f"File format not supported: {path}")


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

        self.zero_lift = 0.0
        
    def _normalize_angle(self, alpha):
        """
        Ensure that the angle fulfils :math:`\\pi < \\alpha < \\pi`

        :param float alpha: Angle in radians
        :return: Normalized angle
        :rtype: float
        """

        return atan2(sin(alpha), cos(alpha))
    
    # ---------- Public API ----------

    def Cl(self, alpha, Re=None):
        """
        Lift coefficient at given angle of attack (in radians) and Reynolds number.
        If multi-Re and Re is None, uses the default_Re (median of available Re).
        If only single-Re data is available, Re is ignored.
        :param float alpha: Angle of attack in radians
        :param float Re: Reynolds number (optional)
        """
        alpha_deg = -self.zero_lift + degrees(self._normalize_angle(alpha))
        if self.re_list:  # multi-Re mode
            use_Re = Re if Re is not None else (self.default_Re or self.re_list[len(self.re_list)//2])
            return self._cl_alpha_re(alpha_deg, use_Re)   # Uses Re dictionary
        # single-Re fallback
        return float(self.Cl_func(alpha_deg))

    def Cd(self, alpha, Re=None):
        """
        Drag coefficient at given angle of attack (in radians) and Reynolds number.
        If multi-Re and Re is None, uses the default_Re (median of available Re).
        If only single-Re data is available, Re is ignored.
        :param float alpha: Angle of attack in radians
        :param float Re: Reynolds number (optional)
        """
        alpha_deg = -self.zero_lift + degrees(self._normalize_angle(alpha))
        if self.re_list:
            use_Re = Re if Re is not None else (self.default_Re or self.re_list[len(self.re_list)//2])
            return self._cd_alpha_re(alpha_deg, use_Re)
        return float(self.Cd_func(alpha_deg))

    # ---------- Internal helpers for multi-Re ----------
    def _cl_alpha_re(self, alpha_deg, Re):
        # exact Re?
        if Re in self._cl_funcs:
            return float(self._cl_funcs[Re](alpha_deg))
        # else interpolate in logRe between neighbors
        return self._interp_in_logRe(alpha_deg, Re, self._cl_funcs)

    def _cd_alpha_re(self, alpha_deg, Re):
        if Re in self._cd_funcs:
            return float(self._cd_funcs[Re](alpha_deg))
        return self._interp_in_logRe(alpha_deg, Re, self._cd_funcs)

    def _interp_in_logRe(self, alpha_deg, Re, func_dict):
        # find neighbors in re_list
        re_sorted = self.re_list
        logRe = np.log(Re)
        logRe_list = np.log(np.array(re_sorted))
        i = bisect_left(re_sorted, Re)

        if i == 0:
            # below smallest: extrapolate linearly in logRe
            f1 = func_dict[re_sorted[0]](alpha_deg)
            f2 = func_dict[re_sorted[1]](alpha_deg)
            w = (logRe - logRe_list[0]) / (logRe_list[1] - logRe_list[0])
            return float((1 - w) * f1 + w * f2)

        if i >= len(re_sorted):
            # above largest: extrapolate
            f1 = func_dict[re_sorted[-2]](alpha_deg)
            f2 = func_dict[re_sorted[-1]](alpha_deg)
            w = (logRe - logRe_list[-2]) / (logRe_list[-1] - logRe_list[-2])
            return float((1 - w) * f1 + w * f2)

        # between i-1 and i
        re1, re2 = re_sorted[i-1], re_sorted[i]
        f1 = func_dict[re1](alpha_deg)
        f2 = func_dict[re2](alpha_deg)
        w = (logRe - np.log(re1)) / (np.log(re2) - np.log(re1))
        return float((1 - w) * f1 + w * f2)

    # ---------- Plot (single-Re or any chosen Re) ----------

    def plot(self, color='k', Re=None, xlim=None):
        """
        If multi-Re and Re provided: plot Cl/Cd vs alpha sampling only
        the valid alpha range for the involved polar(s).
        If xlim=(xmin, xmax) is provided, also restricts to that range.
        """

        # Single-Re case (old behavior)
        if not self.re_list:
            pl.plot(self.alpha_, self.Cl_, color + '-')
            pl.plot(self.alpha_, self.Cd_, color + '--')
            pl.title(f'Airfoil characteristics for {self.name}')
            pl.xlabel('Angle of attack (deg)')
            pl.ylabel('Drag and lift coefficients')
            pl.legend(('$C_l$','$C_d$'))
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
        pl.plot(alpha_grid, Cl_vals, color + '-')
        pl.plot(alpha_grid, Cd_vals, color + '--')
        pl.title(f'Airfoil characteristics for {self.name} @ Re={Re:.0f}')
        pl.xlabel('Angle of attack (deg)')
        pl.ylabel('Drag and lift coefficients')
        pl.legend(('$C_l$', '$C_d$'))
        
        
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
    with_re = []   # lista de (Re, alpha, Cl, Cd)
    defaults = []  # sem Re detectado

    for p in candidates:
        alpha, Cl, Cd, Re = _read_polar_file_with_meta(p)
        if Re is not None:
            with_re.append((Re, alpha, Cl, Cd))
        else:
            defaults.append((alpha, Cl, Cd))

    if with_re:
        # Multi-Re: ordena por Re
        with_re.sort(key=lambda t: t[0])
        a.re_list = [t[0] for t in with_re]
        for Re_val, alpha_deg, Cl, Cd in with_re:
            clf = interp1d(alpha_deg, Cl, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            cdf = interp1d(alpha_deg, Cd, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            a._cl_funcs[Re_val] = clf
            a._cd_funcs[Re_val] = cdf
        a.default_Re = a.re_list[len(a.re_list)//2]  # pick median Re as default
        return a

    # Fallback: single-Re (nenhum arquivo tinha 'Reynolds number' no cabeçalho)
    if defaults:
        alpha_deg, Cl, Cd = defaults[0]
        a.alpha_, a.Cl_, a.Cd_ = alpha_deg, Cl, Cd
        a.Cl_func = interp1d(a.alpha_, a.Cl_, kind='quadratic', bounds_error=False, fill_value='extrapolate')
        a.Cd_func = interp1d(a.alpha_, a.Cd_, kind='quadratic', bounds_error=False, fill_value='extrapolate')
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
    #   python airfoil.py NAME xmin xmax
    #   python airfoil.py NAME xmin xmax Re
    if len(sys.argv) > 3:
        x_lower_limit = float(sys.argv[2])
        x_upper_limit = float(sys.argv[3])
    if len(sys.argv) > 4:
        Re_cli = float(sys.argv[4])

    ax = pl.gca()
    ax.set_xlim(x_lower_limit, x_upper_limit)
    a.plot(Re=Re_cli)
    pl.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate |Cp|min from a cp folder by parsing all CPWR files and saving a CSV with angle, Cp_min, |Cp_min|.

Used when make_cpmin_from_pol.py has been run to generate CPWR files for a range of angles but could not be completed.

Usage examples:
    python make_cpmin_from_cp_folder.py \
        --pol NACA0018_10000.txt \
        --foil-file NACA0018.dat \
        --re 10000 \
        --out-dir run_naca0018_re10000

    python make_cpmin_from_pol.py \
        --pol E63_200000.txt \
        --naca 0018 \
        --re 200000 \
        --out-dir run_naca0018_re200k

Notes:
- Requires XFOIL available in PATH as 'xfoil' or set --xfoil-bin.
- The .pol/.txt is only used to get the angle list; geometry must come from --foil-file or --naca.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from make_cpmin_from_pol import parse_cp_files


def main():
    ap = argparse.ArgumentParser(description="Extract |Cp|min from CPWR files.")
    ap.add_argument("--cp-dir", required=True, help="Path to directory containing CPWR files.")
    ap.add_argument("--airfoil-name", required=True, help="Name of the airfoil (used in output filename).")
    ap.add_argument("--re", type=float, required=True, help="Reynolds number (used in output filename).")
    ap.add_argument("--out-dir", default=".", help="Output directory for CSV file.")
    args = ap.parse_args()

    cp_dir = Path(args.cp_dir)
    airfoil_name = args.airfoil_name
    re = args.re
    out_dir = Path(args.out_dir)
    
    df = parse_cp_files(cp_dir)
    
    file_name = f"cpmin_{airfoil_name}_{int(re)}.csv"
    
    df.to_csv(out_dir / file_name, index=False)
    
    print(f"[✓] Saved CSV: {out_dir / file_name}")


if __name__ == "__main__":
    main()

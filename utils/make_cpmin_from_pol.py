#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate |Cp|min from a .pol/.txt file by:
1) Reading angles from the polar (.pol/.txt),
2) Creating a single XFOIL input script with ALFA+CPWR per angle,
3) Running XFOIL once,
4) Parsing all CPWR files and saving a CSV with angle, Cp_min, |Cp_min|.

Usage examples:
    python make_cpmin_from_pol.py \
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

_rx_alfa = re.compile(r'Alfa\s*=\s*([+-]?\d+(?:\.\d+)?)')

def parse_angles_from_pol(pol_path: Path) -> list[float]:
    """
    Parse angles of attack from an XFOIL .pol file.
    The .pol/.txt typically contains header lines and data lines with columns like:
       Alpha   CL    CD    CDp    CM    Top_Xtr    Bot_Xtr
    We read numeric lines and take the 1st column as Alpha (deg).
    """
    angles = []
    with open(pol_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;":  # comments
                continue
            # skip header lines (contain Alpha, CL, CD, ...)
            if re.search(r"\bAlpha\b", s, flags=re.IGNORECASE):
                continue
            # numeric line? try to parse first token as float
            parts = s.split()
            try:
                alpha = float(parts[0])
            except (ValueError, IndexError):
                continue
            angles.append(alpha)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for a in angles:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique


def write_xfoil_inp(
    inp_path: Path,
    *,
    foil_file: str | None,
    naca_code: str | None,
    angles_deg: list[float],
    re: float,
    mach: float = 0.0,
    iter_max: int = 200,
    n_pan: int = 200,
    t_pan: float = 1.0,
    r_le: float = 0.15,
    vpar_n: int = 5,
    vacc: float = 0.01,
    cp_dir: Path = Path("cp"),
    polar_out: str | None = None,
):
    """
    Create a single XFOIL .inp that:
      - Loads the airfoil (LOAD <file> or NACA <code>),
      - Sets paneling and OPER options,
      - Loops over angles with ALFA + CPWR (<cp_dir>/<file>),
      - Optionally records the polar with PACC <polar_out>.
    """
    cp_dir.mkdir(parents=True, exist_ok=True)
    
    xfoil_source_path = inp_path.parent.parent / 'xfoil.exe'
    xfoil_dest_path = inp_path.parent / 'xfoil.exe'
    shutil.copyfile(xfoil_source_path, xfoil_dest_path)
    
    pplot_source_path = inp_path.parent.parent / 'pplot.exe'
    pplot_dest_path = inp_path.parent / 'pplot.exe'
    shutil.copyfile(pplot_source_path, pplot_dest_path)
    
    pxplot_source_path = inp_path.parent.parent / 'pxplot.exe'
    pxplot_dest_path = inp_path.parent / 'pxplot.exe'
    shutil.copyfile(pxplot_source_path, pxplot_dest_path)

    lines = []
    if foil_file:
        foil_path = Path(foil_file).resolve()
        dst = inp_path.parent / foil_path.name
        if foil_path.resolve() != dst.resolve():
            dst.write_bytes(foil_path.read_bytes())  # copia conteúdo
        
        lines += [f"LOAD {foil_path.name}"]
    elif naca_code:
        lines += [f"NACA {naca_code}"]
    else:
        raise ValueError("Provide either --foil-file or --naca.")

    # Paneling
    lines += [
        "PPAR",
        f"N {n_pan}",
        f"T {t_pan}",
        f"R {r_le}",
        "",
        "",
        "PANE",
        "OPER",
        f"MACH {mach}",
        f"VISC {re:.0f}",
        f"ITER {iter_max}",
        "VPAR",
        f"N {vpar_n}",
        f"VACC {vacc}",
        "",
    ]

    # Optional: record polar while we step through angles
    if polar_out:
        lines += [
            "PACC",
            polar_out,
            "",  # blank line returns to OPER
        ]

    # One CPWR per angle
    for a in angles_deg:
        tag = f"a{a:+06.2f}".replace(".", "_").replace("+", "p").replace("-", "m")
        cp_rel = Path("cp") / f"cp_{tag}.txt"
        lines += [
            f"ALFA {a:.2f}",
            f"CPWR {cp_rel.as_posix()}",
        ]

    # Stop saving polar
    if polar_out:
        lines += [
            "PACC",
            "",
        ]

    lines += [
        "",
        "",
        "QUIT",
        "",
    ]

    inp_path.write_text("\n".join(lines), encoding="utf-8")


def run_xfoil(inp_path: Path, xfoil_bin: str = "xfoil") -> int:
    """Run XFOIL with the given .inp. Returns the process return code."""
    try:
        proc = subprocess.run(
            [xfoil_bin],
            input=inp_path.read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            check=False,
            cwd=inp_path.parent,
        )
    except FileNotFoundError:
        print(f"ERROR: XFOIL binary '{xfoil_bin}' not found in PATH.", file=sys.stderr)
        return 127

    # Save logs next to inp
    inp_dir = inp_path.parent
    (inp_dir / "xfoil_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (inp_dir / "xfoil_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return proc.returncode


def _angle_from_filename(p: Path):
    # fallback: cp_am00_25.txt -> -00.25
    m = re.search(r'cp_a([pm]\d+_\d+)', p.stem)
    if not m:
        return None
    tag = m.group(1)
    s = tag.replace('p', '+').replace('m', '-').replace('_', '.')
    try:
        return float(s)
    except ValueError:
        return None
    
    
def parse_cp_files(cp_dir: Path) -> pd.DataFrame:
    """
    Lê todos os arquivos CPWR em cp_dir (linhas 'x  y  Cp'), encontra min(Cp) por arquivo,
    e retorna DataFrame: file, angle_deg, cp_min, abs_cp_min, n_points.
    """
    rows = []
    for p in sorted(cp_dir.glob("cp_*.txt")):
        cps = []
        alfa = None
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # tenta capturar o ângulo pelo cabeçalho
                m = _rx_alfa.search(s)
                if m:
                    try:
                        alfa = float(m.group(1))
                    except ValueError:
                        pass
                    continue
                # ignora cabeçalhos/comentários
                if s[0].isalpha() or s[0] == '#':
                    continue
                # notação Fortran (D) -> E
                s = s.replace('D', 'E').replace('d', 'e')
                parts = s.split()
                # precisa de pelo menos 3 colunas: x, y, Cp
                if len(parts) >= 3:
                    try:
                        # terceira coluna é Cp
                        cp = float(parts[2])
                        cps.append(cp)
                    except ValueError:
                        continue

        if not cps:
            continue

        if alfa is None:
            alfa = _angle_from_filename(p)

        cp_min = float(np.min(cps))
        rows.append({
            "angle_deg": alfa,
            "cp_min": cp_min,
            "abs_cp_min": abs(cp_min),
            "n_points": len(cps),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["angle_deg"], ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract |Cp|min from XFOIL using angles from a .pol/.txt file.")
    ap.add_argument("--pol", required=True, help="Path to .pol/.txt file (used to read angles).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--foil-file", help="Path to airfoil geometry .dat to LOAD in XFOIL.")
    g.add_argument("--naca", help='NACA code, e.g., "0018". Will run `NACA 0018` in XFOIL.')
    ap.add_argument("--re", type=float, required=True, help="Reynolds number for VISC.")
    ap.add_argument("--mach", type=float, default=0.0, help="Mach number (default 0).")
    ap.add_argument("--iter", type=int, default=200, help="Max iterations in OPER/ITER.")
    ap.add_argument("--out-dir", default="xfoil_cp_run", help="Output directory (inp, logs, cp/, csv).")
    ap.add_argument("--xfoil-bin", default="xfoil", help="XFOIL binary name/path.")
    ap.add_argument("--save-polar", action="store_true", help="Also PACC a new polar during the run.")
    # Paneling/visc options (tune as needed)
    ap.add_argument("--n-pan", type=int, default=200, help="PPAR N")
    ap.add_argument("--t-pan", type=float, default=1.0, help="PPAR T")
    ap.add_argument("--r-le", type=float, default=0.15, help="PPAR R")
    ap.add_argument("--vpar-n", type=int, default=5, help="VPAR N")
    ap.add_argument("--vacc", type=float, default=0.01, help="VPAR VACC")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        # Keep it simple; you can change to overwrite logic if desired
        print(f"[i] Output directory exists: {out_dir}. Files may be overwritten.", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)

    pol_path = Path(args.pol)
    if not pol_path.is_file():
        print(f"ERROR: .pol/.txt not found: {pol_path}", file=sys.stderr)
        sys.exit(2)

    angles = parse_angles_from_pol(pol_path)
    if not angles:
        print("ERROR: No angles found in .pol.txt file.", file=sys.stderr)
        sys.exit(3)

    print(f"[i] Found {len(angles)} angles in polar: min={min(angles):.2f}°, max={max(angles):.2f}°")

    cp_dir = out_dir / "cp"
    inp_path = out_dir / "run_xfoil_cp.inp"
    polar_new = (out_dir / (pol_path.stem + "_replay.txt")).as_posix() if args.save_polar else None

    write_xfoil_inp(
        inp_path=inp_path,
        foil_file=args.foil_file,
        naca_code=args.naca,
        angles_deg=angles,
        re=args.re,
        mach=args.mach,
        iter_max=args.iter,
        n_pan=args.n_pan,
        t_pan=args.t_pan,
        r_le=args.r_le,
        vpar_n=args.vpar_n,
        vacc=args.vacc,
        cp_dir=cp_dir,
        polar_out=polar_new,
    )

    print(f"[i] Running XFOIL with input: {inp_path}")
    rc = run_xfoil(inp_path, xfoil_bin=args.xfoil_bin)
    if rc != 0:
        print(f"ERROR: XFOIL exited with code {rc}. Check logs in {out_dir}", file=sys.stderr)
        sys.exit(rc)

    print(f"[i] Parsing CPWR files from: {cp_dir}")
    df = parse_cp_files(cp_dir)
    if df.empty:
        print("ERROR: No CP files parsed. Check XFOIL logs and input.", file=sys.stderr)
        sys.exit(4)

    if args.foil_file:
        file_name_parts = args.foil_file.split('.')
        out_csv = out_dir / f"cpmin_{file_name_parts[0]}_{int(args.re)}.csv"
    elif args.naca:
        out_csv = out_dir / f"cpmin_naca{args.naca}_{int(args.re)}.csv"
    else:
        raise ValueError("Provide either --foil-file or --naca.")

    
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved CSV: {out_csv}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

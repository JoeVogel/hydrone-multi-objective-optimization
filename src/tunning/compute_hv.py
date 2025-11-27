# compute_hv.py

import pandas as pd
from pathlib import Path
import numpy as np
import sys

def get_latest_pareto_front():
    # results/ fica ao lado de optimization/
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError("No run directories found inside results/")

    # nomes são timestamps YYYY-MM-DD_HH-MM-SS → sort lexicograficamente funciona
    latest = sorted(run_dirs)[-1]
    pareto_path = latest / "pareto_front.csv"
    if not pareto_path.exists():
        raise FileNotFoundError(f"pareto_front.csv not found in {latest}")

    return pareto_path

def compute_hv_2d(points, ref=(0.0, 0.0)):
    """
    points: array shape (N, 2), objetivos a maximizar.
    ref: ponto de referência "pior" (assumimos 0,0).
    """
    if points.size == 0:
        return 0.0

    # Clipa em relação ao ref para ter tudo >= ref
    f1 = np.maximum(points[:, 0], ref[0])
    f2 = np.maximum(points[:, 1], ref[1])

    # Remove pontos com qualquer NaN
    mask = ~np.isnan(f1) & ~np.isnan(f2)
    f1, f2 = f1[mask], f2[mask]

    if f1.size == 0:
        return 0.0

    # Ordena por f1 crescente
    order = np.argsort(f1)
    f1 = f1[order]
    f2 = f2[order]

    hv = 0.0
    prev_f1 = ref[0]

    # Em frente de Pareto para max, f2 deve ser não crescente com f1 crescente.
    # Se não for exatamente, ainda assim essa fórmula é uma aproximação razoável.
    for x, y in zip(f1, f2):
        width = x - prev_f1
        height = max(y - ref[1], 0.0)
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = x

    return hv

def main():
    try:
        pareto_path = get_latest_pareto_front()
        df = pd.read_csv(pareto_path)

        if "aerial_fitness" not in df.columns or "aquatic_fitness" not in df.columns:
            raise ValueError("Columns 'aerial_fitness' and 'aquatic_fitness' not found.")

        pts = df[["aerial_fitness", "aquatic_fitness"]].to_numpy(dtype=float)

        # Clipa negativos para 0 (não queremos HV negativo)
        pts = np.maximum(pts, 0.0)

        hv = compute_hv_2d(pts, ref=(0.0, 0.0))

        # irace MINIMIZA, então retornamos -HV
        print(-hv)
    except Exception as e:
        # Se der erro, devolvemos um valor bem ruim (grande positivo)
        print(1e9, file=sys.stderr)
        print(1e9)

if __name__ == "__main__":
    main()

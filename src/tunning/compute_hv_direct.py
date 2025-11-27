import sys
import json
import numpy as np

def compute_hv_2d(points, ref=(0.0, 0.0)):
    if points.size == 0:
        return 0.0

    f1 = np.maximum(points[:, 0], ref[0])
    f2 = np.maximum(points[:, 1], ref[1])

    mask = ~np.isnan(f1) & ~np.isnan(f2)
    f1, f2 = f1[mask], f2[mask]
    if f1.size == 0:
        return 0.0

    order = np.argsort(f1)
    f1 = f1[order]
    f2 = f2[order]

    hv = 0.0
    prev_f1 = ref[0]
    for x, y in zip(f1, f2):
        width = x - prev_f1
        height = max(y - ref[1], 0.0)
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = x

    return hv

def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(1e9)
            return

        data = json.loads(raw)
        arr = np.array(data, dtype=float)

        hv = compute_hv_2d(arr, ref=(0.0, 0.0))
        print(-float(hv))  # irace minimiza
    except Exception:
        print(1e9)

if __name__ == "__main__":
    main()

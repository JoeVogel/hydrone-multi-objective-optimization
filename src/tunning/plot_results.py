import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Caminho absoluto para a pasta onde está o script
script_dir = Path(__file__).resolve().parent

# results está no mesmo nível de tunning → ../results
base_path = script_dir.parent / "results"

timestamps = []
num_rows_list = []

# Itera pelas subpastas
for folder in sorted(base_path.iterdir()):
    if folder.is_dir():

        # Extrai o timestamp do nome da pasta
        timestamp = folder.name.strip()
        csv_path = folder / "pareto_front.csv"

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                num_rows = len(df)
            except Exception as e:
                print(f"Erro ao ler {csv_path}: {e}")
                continue

            timestamps.append(timestamp)
            num_rows_list.append(num_rows)

# Converter timestamps para datetime e ordenar
ts = pd.to_datetime(timestamps, format="%Y-%m-%d_%H-%M-%S")
order = ts.argsort()

ts = ts[order]
num_rows_list = [num_rows_list[i] for i in order]

# Plot com Matplotlib
plt.figure(figsize=(12, 6))
plt.plot(ts, num_rows_list, marker='o', linestyle='-', linewidth=2)

plt.title("Número de linhas do pareto_front.csv por execução", fontsize=14)
plt.xlabel("Timestamp da execução")
plt.ylabel("Quantidade de linhas")
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()

plt.show(block=True)

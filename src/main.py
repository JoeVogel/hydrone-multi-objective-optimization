import logging
import tomllib
import os

# Configurar o log para imprimir no console
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato da saída
)

logger = logging.getLogger(__name__)

from scenario                   import Scenario
from rotor                      import Rotor
from evaluation.aerial_methods  import AerialBEMT
from evaluation.aquatic_methods import WaterBEMT
from optimization.nsgaii        import NSGAII

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

def load_config_toml(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)

if __name__ == "__main__":


    # Load configurations
    config_base_dir = Path(__file__).resolve().parent.parent
    config_path = config_base_dir / "data" / "configuration.toml"
    configs = load_config_toml(config_path)
    
    # Initialize evaluation methods
    aerial_evaluator = AerialBEMT(
        scenario=Scenario(rpm=4000.0, v_inf=0.0)
    )

    # 3700 RPM mantains tip speed below 39 m/s for 20 cm diameter propeller
    aquatic_evaluator = WaterBEMT(
        scenario=Scenario(rpm=400.0, v_inf=0.0)
    )
    
    optimizer = NSGAII(configs, aerial_evaluator, aquatic_evaluator)
    
    pareto_fronts = optimizer.run()

    # Plot Pareto fronts
    plt.figure(figsize=(9, 6))

    cmap = cm.get_cmap("tab10")  # paleta com bom contraste
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]  # alterna marcadores

    show_all_fronts = False

    if show_all_fronts:
        for i, front in enumerate(pareto_fronts):
            x = np.array([ind.aerial_fitness for ind in front])
            y = np.array([ind.aquatic_fitness for ind in front])

            order = np.argsort(x)
            x_sorted, y_sorted = x[order], y[order]

            color = cmap(i % cmap.N)
            marker = markers[i % len(markers)]

            plt.plot(x_sorted, y_sorted, lw=1.5, alpha=0.7, color=color)
            plt.scatter(x, y, s=60, marker=marker,
                        facecolors=color, edgecolors="black", linewidths=0.6,
                        alpha=0.9, label=f"Front {i+1}")
    else:
        f1 = pareto_fronts[0]
        x = np.array([ind.aerial_fitness for ind in f1])
        y = np.array([ind.aquatic_fitness for ind in f1])

        order = np.argsort(x)
        x_sorted, y_sorted = x[order], y[order]

        plt.plot(x_sorted, y_sorted, lw=1.5, alpha=0.7, color="blue")
        plt.scatter(x, y, s=60, marker="o",
                    facecolors="blue", edgecolors="black", linewidths=0.6,
                    alpha=0.9, label="Front 1")

    plt.title("Frentes de Pareto — NSGA-II", pad=10)
    plt.xlabel("Fitness Aéreo (ηₐ)")
    plt.ylabel("Fitness Aquático (η𝑤)")
    plt.grid(True, which="both", ls=":", lw=0.8, alpha=0.6)

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.show()

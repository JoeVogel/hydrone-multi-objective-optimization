import logging
import tomllib
import json
import argparse

# Configurar o log para imprimir no console
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato da saída
)

logger = logging.getLogger(__name__)

from scenario                   import Scenario
from evaluation.aerial_methods  import AerialBEMT
from evaluation.aquatic_methods import WaterBEMT
from optimization.nsgaii        import NSGAII
from motor                      import Motor
from fluid                      import Fluid, FluidType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

def load_config_toml(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)

def get_var(variables_list, name):
    for v in variables_list:
        if v["name"] == name:
            return v
    raise KeyError(f"Variable '{name}' not found in configuration")

def plot_fronts(pareto_fronts, show_all_fronts=False):
    # Plot Pareto fronts
    plt.figure(figsize=(9, 6))

    cmap = cm.get_cmap("tab10")  # paleta com bom contraste
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]  # alterna marcadores

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

if __name__ == "__main__":
    
    # ---------- CLI arguments ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--elitism_fraction", type=float, default=None)
    parser.add_argument("--mutation_rate", type=float, default=None)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no_plot",
        action="store_false",
        dest="plot",
        help="Disable plotting of Pareto fronts"
    )
    parser.set_defaults(plot=True)

    parser.add_argument(
        "--no_write_log_file",
        action="store_false",
        dest="write_log_file",
        help="Disable writing log file"
    )
    parser.set_defaults(write_log_file=True)

    args = parser.parse_args()

    # Load configurations
    config_base_dir = Path(__file__).resolve().parent.parent
    config_path = config_base_dir / "data" / "configuration.toml"
    configs = load_config_toml(config_path)
    
    variables = configs["variables"]
    
    alpha       = get_var(variables, "alpha")
    blades      = get_var(variables, "n_blades")
    foil_list   = get_var(variables, "airfoil_list")
    
    problem_config = {
        "diameter"              : configs["problem"]["diameter"],
        "number_of_sections"    : configs["problem"]["number_of_sections"],
        "hub_radius"            : configs["problem"]["hub_radius"],
        "max_chord_global"      : configs["problem"]["max_chord_global"] if "max_chord_global" in configs["problem"] else None,
        "min_alpha"             : alpha["min"], 
        "max_alpha"             : alpha["max"],
        "min_blade_number"      : blades["min"],
        "max_blade_number"      : blades["max"],
        "foil_options"          : foil_list["choices"],
    }
    
    nsga_elite      = configs["nsga2"].get("elite_fraction", 0.5)
    nsga_mutrate    = configs["nsga2"].get("mutation_rate", 0.1)
    nsga_popsize    = configs["nsga2"].get("pop_size", 60)
    nsga_gener      = configs["nsga2"].get("generations", 60)
    nsga_seed       = configs["nsga2"].get("seed", 42)
    
    # Overrides from CLI arguments
    if args.elitism_fraction is not None:
        nsga_elite = args.elitism_fraction
    if args.mutation_rate is not None:
        nsga_mutrate = args.mutation_rate
    if args.pop_size is not None:
        nsga_popsize = args.pop_size
    if args.generations is not None:
        nsga_gener = args.generations
    if args.seed is not None:
        nsga_seed = args.seed

    write_log_file = args.write_log_file

    nsga_config = {
        "population_size": nsga_popsize,
        "maximum_generations": nsga_gener,
        "seed": nsga_seed,
        "elitism_fraction": nsga_elite,
        "mutation_rate": nsga_mutrate,
    }

    motor = Motor(
        KV=configs["motor"]["KV"], 
        voltage=configs["motor"]["voltage"], 
        max_current=configs["motor"]["max_current"]
    )

    aerial_rpm = 3000.0  

    # motor_data = {
    #     "aerial_Q_max": motor.torque_available(aerial_rpm),
    #     "aquatic_Q_max": motor.torque_available(aquatic_rpm),
    # }
        
    # Initialize evaluation methods
    aerial_evaluator = AerialBEMT(
        scenario=Scenario(rpm=aerial_rpm, v_inf=0.0)
    )

    rho1 = Fluid(FluidType.AIR).rho
    rho2 = Fluid(FluidType.WATER).rho

    aquatic_rpm = aerial_rpm * ((rho1 / rho2) ** (1/3))

    aquatic_evaluator = WaterBEMT(
        scenario=Scenario(rpm=int(aquatic_rpm), v_inf=0.0)
    )
    
    optimizer = NSGAII(aerial_evaluator, aquatic_evaluator, problem_config, nsga_config, write_log_file)
    
    pareto_fronts = optimizer.run()

    if args.plot == True:
        plot_fronts(pareto_fronts)
    
    # O NSGA-II retorna uma lista de frentes.
    # A primeira frente é pareto_fronts[0].
    front = pareto_fronts[0]

    # Criar lista de pares [aerial_fitness, aquatic_fitness]
    pareto = [
        [float(ind.aerial_fitness), float(ind.aquatic_fitness)]
        for ind in front
    ]

    # Importante: NÃO imprimir nada além disso quando rodando no Irace
    print(json.dumps(pareto))

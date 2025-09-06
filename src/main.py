import logging

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

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # Initialize evaluation methods
    aerial_evaluator = AerialBEMT(
        scenario=Scenario(rpm=4000.0, v_inf=3.0)
    )

    # 3700 RPM mantains tip speed below 39 m/s for 20 cm diameter propeller
    aquatic_evaluator = WaterBEMT(
        scenario=Scenario(rpm=400.0, v_inf=0.3)
    )
    
    optimizer = NSGAII(30, 30, aerial_evaluator, aquatic_evaluator)
    
    pareto_fronts = optimizer.run()

    # Plot Pareto fronts
    plt.figure(figsize=(8, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']  # Cores para os fronts

    all_fronts = False

    if all_fronts:

        for i, front in enumerate(pareto_fronts):
            x = [ind.aerial_fitness for ind in front]   # Fitness Aéreo (X)
            y = [ind.aquatic_fitness for ind in front]  # Fitness Aquático (Y)
            
            plt.scatter(x, y, color=colors[i % len(colors)], label=f'Front {i+1}', alpha=0.7, edgecolors='k')
            plt.title("Pareto Fronts - NSGA-II Optimization")
    else:
        
        # Extrai os valores de fitness para os eixos X e Y
        x = [ind.aerial_fitness for ind in pareto_fronts[0]]   # Fitness Aéreo (X)
        y = [ind.aquatic_fitness for ind in pareto_fronts[0]]  # Fitness Aquático (Y)

        plt.scatter(x, y, color='red', label='Front Pareto (F1)', alpha=0.8, edgecolors='k')
        plt.title("Pareto Front - NSGA-II Optimization")
        
    plt.xlabel("Fitness Aéreo")
    plt.ylabel("Fitness Aquático")
    
    plt.legend()
    plt.grid(True)
    plt.show()


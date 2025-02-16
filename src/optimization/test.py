import logging

# Configurar o log para imprimir no console
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato da saída
)

logger = logging.getLogger(__name__)

from evaluation.aerial_methods      import AerialBEMT
from evaluation.aquatic_methods     import WaterBEMT
from optimization.nsgaii            import NSGAII

import matplotlib.pyplot as plt

if __name__ == "__main__":

    alpha_values = [1, 3, 5, 7]  # Angulo de ataque
    D_values = [0.2, 0.25, 0.3]  # Diâmetros da hélice (m)
    B_values = [2, 3]  # Número de pás
    
    aerial_evaluator = AerialBEMT()
    aquatic_evaluator = WaterBEMT()
    
    # Loop para calcular para diferentes alpha, D e B
    for a in alpha_values:
        for D in D_values:
            for B in B_values:
                T, Q, eta, cav, v_tip, re = aquatic_evaluator.evaluate(a, D, B)
                print(f"alpha: {a:.2f}, D: {D:.2f}m, B: {B}, Thrust: {T:.4f} N, Torque: {Q:.4f} Nm, Efficiency: {eta:.4f}, Cavitation: {cav:.4f}")



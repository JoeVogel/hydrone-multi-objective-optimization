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
    
    aerial_evaluator = AerialBEMT()
    aquatic_evaluator = WaterBEMT()
    
    a = 5.0
    D = 0.15
    B = 2

    print("HYDRODYNAMICS")
    T, Q, eta, cav, v_tip, re = aquatic_evaluator.evaluate(a, D, B, n=500)
    print(f"alpha: {a:.2f}º, D: {D:.2f}m, B: {B} blades, Thrust: {T:.4f} N, Torque: {Q:.4f} Nm, Efficiency: {eta:.4f}, Tip speed: {v_tip:.1f} m/s, Cavitation: {cav:.4f}, Re: {re}")

    print("AERODYNAMICS")
    T, Q, eta, re = aerial_evaluator.evaluate(a, D, B, n=5000)
    print(f"alpha: {a:.2f}º, D: {D:.2f}m, B: {B} blades, Thrust: {T:.4f} N, Torque: {Q:.4f} Nm, Efficiency: {eta:.4f}, Re: {re}")
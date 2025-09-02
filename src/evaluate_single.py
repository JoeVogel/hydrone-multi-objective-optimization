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

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    diameter= 0.2
    radius_hub = 0.0025
    n_blades= 2
    alpha=12
    number_of_sections = 5

    rotor = Rotor(
        n_blades=n_blades,
        diameter=diameter,
        radius_hub=radius_hub,
        number_of_sections=number_of_sections,
        foil_list=['NACA_4412'] * number_of_sections,
        chord_list=[0.0152, 0.0136, 0.0120, 0.0104, 0.0088],
        pitch_list=[alpha] * number_of_sections
    )

    # Evaluate in air
    air_solver = AerialBEMT(
        scenario=Scenario(rpm=1100.0, v_inf=1.0)
    )

    T, Q, P, eta = air_solver.evaluate(rotor)

    print("--- Results ---")
    print("")
    print("Evaluation Type: ", air_solver.type)
    print("Thrust: ", T, "N")
    print("Torque: ", Q, "Nm")
    print("Power: ", P, "W")     
    print("Efficiency: ", eta) 
    print("")

    # Evaluate in water
    watter_solver = WaterBEMT(
        scenario=Scenario(rpm=180.0, v_inf=1.0)
    )

    T, Q, P, eta = watter_solver.evaluate(rotor)     

    print("Evaluation Type: ", watter_solver.type)
    print("Thrust: ", T, "N")
    print("Torque: ", Q, "Nm")          
    print("Power: ", P, "W")   
    print("Efficiency: ", eta) 
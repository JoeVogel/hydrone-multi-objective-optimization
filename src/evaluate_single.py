import logging
import math

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

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

if __name__ == "__main__":
    
    diameter= 0.2
    radius_hub = 0.0025
    n_blades= 2
    alpha=12
    number_of_sections = 20

    rotor = Rotor(
        n_blades=n_blades,
        diameter=diameter,
        radius_hub=radius_hub,
        number_of_sections=number_of_sections,
        foil_list=['NACA0018'] * number_of_sections,
        chord_list = [
            0.01700, 0.01660, 0.01620, 0.01580, 0.01540,
            0.01500, 0.01440, 0.01380, 0.01320, 0.01260,
            0.01220, 0.01180, 0.01140, 0.01100, 0.01060,
            0.01020, 0.00980, 0.00940, 0.00900, 0.00860
            ],
        pitch_list=[alpha] * number_of_sections
    )

    """
        To evaluate the propeller, we need to define the operating conditions,
        which include the rotational speed (RPM) and the inflow velocity (V_inf).
        These parameters can be chosen based on the desired advance ratio (J),
        which is a dimensionless parameter that relates the forward speed of the
        propeller to its rotational speed and diameter.
        The advance ratio is defined as:
    
        J = V_inf / (n * D)
        where:
        J = advance ratio (dimensionless)
        V_inf = inflow velocity (m/s)
        n = rotational speed (revolutions per second, RPS = RPM/60)
        D = diameter of the propeller (m)

        Typical values for J:
        - For hover: J = 0 (V_inf = 0)
        - For low-speed forward flight: J = 0.2 to 0.5
        - For high-speed forward flight: J = 0.5 to 1.0

        Example:
        Let's assume we want to evaluate the propeller at a forward speed of 2 m/s
        and we want to find the RPM that gives us a J value of 0.3
        for a propeller with a diameter of 0.2 m.
        D = 0.2 m
        V_inf = 2 m/s
        J = 0.3
        n = V_inf / (J * D) = 2 / (0.3 * 0.2) = 33.33 RPS = 2000 RPM
    """

    # Evaluate in air
    air_solver = AerialBEMT(
        scenario=Scenario(rpm=2127.0, v_inf=0.0)
    )

    T, Q, P, J, CT, CQ, CP, eta = air_solver.evaluate(rotor)

    print("--- Results ---")
    print("")
    print("Evaluation Type: ", air_solver.type)
    print("Thrust: ", T, "N")
    print("Torque: ", Q, "Nm")
    print("Power: ", P, "W")     
    print("Advance Ratio: ", J)
    print("Thrust Coefficient: ", CT)
    print("Torque Coefficient: ", CQ)
    print("Power Coefficient: ", CP)
    print("Efficiency: ", eta) 
    print("")

    # For fair comparison, we use rpm and v_inf in watter that generates the same J used in air

    # Evaluate in water
    watter_solver = WaterBEMT(
        scenario=Scenario(rpm=400.0, v_inf=0.0)
    )

    T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion = watter_solver.evaluate(rotor)     

    print("Evaluation Type: ", watter_solver.type)
    print("Thrust: ", T, "N")
    print("Torque: ", Q, "Nm")          
    print("Power: ", P, "W")   
    print("Advance Ratio: ", J)
    print("Thrust Coefficient: ", CT)
    print("Torque Coefficient: ", CQ)
    print("Power Coefficient: ", CP)
    print("Efficiency: ", eta) 
    print("Cavitating Proportion", cavitating_proportion)
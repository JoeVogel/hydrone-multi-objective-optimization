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

if __name__ == "__main__":
    
    # diameter= 0.2
    # radius_hub = 0.0025
    # n_blades= 2
    # alpha=12
    # number_of_sections = 20

    # rotor = Rotor(
    #     n_blades=n_blades,
    #     diameter=diameter,
    #     radius_hub=radius_hub,
    #     number_of_sections=number_of_sections,
    #     foil_list=['NACA0018'] * number_of_sections,
    #     chord_list = [
    #         0.01700, 0.01660, 0.01620, 0.01580, 0.01540,
    #         0.01500, 0.01440, 0.01380, 0.01320, 0.01260,
    #         0.01220, 0.01180, 0.01140, 0.01100, 0.01060,
    #         0.01020, 0.00980, 0.00940, 0.00900, 0.00860
    #         ],
    #     pitch_list=[alpha] * number_of_sections
    # )

    # Geometria principal (do arquivo do 10x4.5 MR)
    diameter     = 0.254      # 10 in
    radius_hub   = 0.02032    # 0.8 in  (HUBTRA)
    n_blades     = 2

    # Estações radiais (m) — do arquivo (0.80" ... 5.00")
    stations_m = [
        0.020320, 0.021844, 0.023368, 0.024892, 0.026416, 0.027940, 0.029464, 0.031118,
        0.034011, 0.037033, 0.040056, 0.043078, 0.046100, 0.049122, 0.052144, 0.055166,
        0.058188, 0.061210, 0.064233, 0.067255, 0.070277, 0.073299, 0.076321, 0.079343,
        0.082365, 0.085387, 0.088409, 0.091431, 0.094454, 0.097476, 0.100498, 0.103520,
        0.106542, 0.109564, 0.112586, 0.115608, 0.118630, 0.121653, 0.124675, 0.127000
    ]

    # Corda (m) — do arquivo
    chord_list = [
        0.018674, 0.019776, 0.020813, 0.021781, 0.022682, 0.023515, 0.02428, 0.025037, 
        0.02617, 0.027099, 0.02777, 0.028184, 0.028344, 0.02825, 0.028011, 0.027711, 
        0.027356, 0.026944, 0.026479, 0.025964, 0.025397, 0.02478, 0.02412, 0.023414, 
        0.022664, 0.021877, 0.021049, 0.020183, 0.019284, 0.018349, 0.017386, 0.016391, 
        0.01537, 0.014323, 0.013251, 0.012156, 0.011044, 0.009728, 0.007127, 0.000003
    ]

    # Pitch PRATHER (polegadas) -> ângulo local (graus) por seção
    pitch_pr_in = [
        3.1671, 3.5392, 3.8486, 4.0903, 4.2613, 4.3597, 4.3918, 4.4057,
        4.4226, 4.4329, 4.4370, 4.4374, 4.4365, 4.4352, 4.4332, 4.4309,
        4.4283, 4.4254, 4.4224, 4.4192, 4.4156, 4.4116, 4.4071, 4.4019,
        4.3960, 4.3891, 4.3811, 4.3719, 4.3612, 4.3488, 4.3347, 4.3185,
        4.3003, 4.2797, 4.2567, 4.2310, 4.2022, 4.1701, 4.1348, 4.5147
    ]
    inch = 0.0254
    pitch_list = [
        math.degrees(math.atan2(p*inch, 2*math.pi*r))
        for p, r in zip(pitch_pr_in, stations_m)
    ]

    # Aerofólios por seção (E63 até 2.85"; NACA4412 a partir de 4.65"; zona de transição tratada como NACA4412)
    r_e63_end    = 2.85 * inch
    r_apc12_beg  = 4.65 * inch
    foil_list = [
        'E63' if r <= r_e63_end else ('NACA4412' if r >= r_apc12_beg else 'NACA4412')
        for r in stations_m
    ]

    number_of_sections = len(stations_m)

    rotor = Rotor(
        n_blades=n_blades,
        diameter=diameter,
        radius_hub=radius_hub,
        number_of_sections=number_of_sections,
        foil_list=foil_list,
        chord_list=chord_list,
        pitch_list=pitch_list,
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
        scenario=Scenario(rpm=4000.0, v_inf=0.0)
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
    # watter_solver = WaterBEMT(
    #     scenario=Scenario(rpm=400.0, v_inf=0.0)
    # )

    # T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion = watter_solver.evaluate(rotor)     

    # print("Evaluation Type: ", watter_solver.type)
    # print("Thrust: ", T, "N")
    # print("Torque: ", Q, "Nm")          
    # print("Power: ", P, "W")   
    # print("Advance Ratio: ", J)
    # print("Thrust Coefficient: ", CT)
    # print("Torque Coefficient: ", CQ)
    # print("Power Coefficient: ", CP)
    # print("Efficiency: ", eta) 
    # print("Cavitating Proportion", cavitating_proportion)
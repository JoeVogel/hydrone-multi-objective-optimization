# -*- coding: utf-8 -*-

"""
Module for holding and calculating fluid properties. Currently, only viscosity and density are included.
"""

from enum import Enum


class FluidType (Enum):
    """
    Enum for different fluid types.
    """
    AIR = "air"
    WATER = "water"

class Fluid:
    """
    Class for loading fluid properties from configuration file and providing them to the solver.

    :param type: Type of fluid (FluidType.AIR or FluidType.WATER)
    """

    def __init__(self, type: FluidType = FluidType.AIR):

        # Kinematic viscosity (m²/s) is the ratio between dynamic viscosity (kg/m·s) and density (kg/m³).
        # kinematic viscosity = dynamic viscosity / density

        # rho = density (kg/m³)
        # mu = dynamic viscosity (kg/m·s)
        # nu = kinematic viscosity (m²/s)
        # pv = vapor pressure (kPa) (Antoine)

        if (type == FluidType.AIR):
            self.rho = 1.225  
            self.mu = 1.81e-5  
        elif (type == FluidType.WATER):
            self.rho = 991.0  
            self.mu = 1.138e-3 
            self.pv = 1.706
        else:
            raise ValueError(f"Unsupported fluid type: {type}")
        
        self.nu = self.mu / self.rho
        self.temperature = 15.0



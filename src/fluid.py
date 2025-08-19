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

        if (type == FluidType.AIR):
            self.rho = 1.225  # Densidade do ar (kg/m³)
            self.mu = 1.81e-5  # Viscosidade dinâmica do ar (kg/m.s)
        elif (type == FluidType.WATER):
            self.rho = 997.0  # Densidade da água (kg/m³)
            self.mu = 8.9e-4  # Viscosidade dinâmica da água (kg/m.s)
        else:
            raise ValueError(f"Unsupported fluid type: {type}")



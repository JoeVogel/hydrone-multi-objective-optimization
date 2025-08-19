"# -*- coding: utf-8 -*-"

"""
Module for the scenario class.
"""

class Scenario:
    """
    Class for loading scenario properties from configuration file and providing them to the solver.

    :param float rpm: RPM of the propeller
    :param float v_inf: Freestream velocity in m/s  
    :param float twist: Twist angle in degrees (default is 0.0)  
    """
    
    def __init__(self, rpm, v_inf):
        self.rpm = 1100.0  # RPM of the propeller
        self.v_inf = 1.0  # Freestream velocity in m/s
        self.twist = 0.0  # Twist angle in degrees
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
    
    def __init__(self, rpm, v_inf, twist=0.0):
        self.rpm = rpm
        self.v_inf = v_inf
        self.twist = twist 
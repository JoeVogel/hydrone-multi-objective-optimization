# -*- coding: utf-8 -*-
"""
Module for the solver class.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import optimize
from configparser import ConfigParser
from math import radians, degrees, sqrt, cos, sin, atan2, atan, pi, acos, exp
from fluid import Fluid
from rotor import Rotor
from scenario import Scenario


class Solver: 
    """
    The Solver object gets the configs and contains functions for running a single simulation,
    parameter sweeps and optimization.

    """
    def __init__(self, fluid:Fluid, scenario:Scenario):
        
        # Scenario
        self.scenario = scenario

        # Fluid
        self.fluid = fluid

        # Solver
        self.solver = 'bisect'  # Default solver method, can be 'brute' or 'bisect'

        # Output
        self.T = 0 # Thrust
        self.Q = 0 # Torque
        self.P = 0 # Power
  
    def rotor_coeffs(self, T, Q, P):
        """ 
        Dimensionless coefficients for a rotor. 

        .. math::
            \\text{J} = \\frac{V_\\infty}{nD} \\\\
            C_T = \\frac{T}{\\rho n^2 D^4} \\\\
            C_Q = \\frac{Q}{\\rho n^2 D^5} \\\\
            C_P = 2\\pi C_Q \\\\
            \\eta = \\frac{C_T}{C_P}J \\\\

        :param float T: Thrust
        :param float Q: Torque
        :param float P: Power
        :return: Advance ratio, thrust coefficient, torque coefficient, power coefficient and efficiency
        :rtype: tuple
        """

        D = self.rotor.diameter
        R = 0.5*D
        rho = self.fluid.rho
        n = self.scenario.rpm/60.0
        J = self.scenario.v_inf/(n*D)
        omega = self.scenario.rpm*2*pi/60.0
 
        CT = T/(rho*n**2*D**4)
        CQ = Q/(rho*n**2*D**5)
        CP = 2*pi*CQ

        if J==0.0:
            eta = (CT/CP)
        else:
            eta = (CT/CP)*J

        return J, CT, CQ, CP, eta

    def solve(self, rotor, scenario):
        """
        Find inflow angle and calculate forces for a single rotor given rotational speed, inflow velocity and radius.

        :rtype: tuple
        """

        rotor.precalc(scenario.twist)

        omega = scenario.rpm*2*pi/60.0
        # Axial momentum (thrust)
        T = 0.0
        # Angular momentum
        Q = 0.0

        for sec in rotor.sections:

            if sec.radius < rotor.blade_radius:
                v = scenario.v_inf
            else:
                v = 0.0
                
            if self.solver == 'brute':
                phi = self.brute_solve(sec, v, omega)
            else:
                try:
                    phi = optimize.bisect(sec.func, 0.01*pi, 0.9*pi, args=(v, omega))
                except ValueError as e:
                    print(e)
                    print('Bisect failed, switching to brute solver')
                    phi = self.brute_solve(sec, v, omega)

            
            dT, dQ = sec.forces(phi, v, omega, self.fluid)

            # Integrate
            T += dT
            Q += dQ
        
        # Power
        P = Q*omega  

        return T, Q, P
 
    def run(self, rotor:Rotor):
        """
        Runs the solver, i.e. finds the forces for each rotor.

        :return: Calculated thrust, torque, power and DataFrame with properties for all sections.
        :rtype: tuple
        """
        self.rotor = rotor
        self.T, self.Q, self.P = self.solve(self.rotor, self.scenario)
        
        return self.T, self.Q, self.P, self.rotor.sections_dataframe()

    def brute_solve(self, sec, v, omega, n=3600):
        """ 
        Solve by a simple brute force procedure, iterating through all
        possible angles and selecting the one with lowest residual.

        :param Section sec: Section to solve for
        :param float v: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :param int n: Number of angles to test for, optional
        :return: Inflow angle with lowest residual
        :rtype: float
        """
        resid = np.zeros(n)
        phis = np.linspace(-0.9*np.pi,0.9*np.pi,n)
        for i,phi in enumerate(phis):
            res = sec.func(phi, v, omega)
            if not np.isnan(res):
                resid[i] = res
            else:
                resid[i] = 1e30
        i = np.argmin(abs(resid))
        return phis[i]
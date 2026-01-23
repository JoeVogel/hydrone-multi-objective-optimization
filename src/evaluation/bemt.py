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

        self.omega = self.scenario.rpm*2*pi/60.0   # Rotational speed in rad/s

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
            J   = V_inf / (n * D)
            C_T = T / (rho * n^2 * D^4)
            C_Q = Q / (rho * n^2 * D^5)
            C_P = 2π * C_Q
            eta = (C_T / C_P) * J

        :param float T: Thrust
        :param float Q: Torque
        :param float P: Power
        :return: Advance ratio, thrust coefficient, torque coefficient, power coefficient and efficiency
        :rtype: tuple
        """

        D = self.rotor.diameter
        R = 0.5*D
        rho = self.fluid.rho
        n = self.scenario.rpm/60.0 #RPS
        J = self.scenario.v_inf/(n*D)
 
        CT = T/(rho*n**2*D**4)
        CQ = Q/(rho*n**2*D**5)
        CP = 2*pi*CQ

        if J==0.0:
            eta = 0.0
        else:
            eta = (CT/CP)*J

        return J, CT, CQ, CP, eta

    def solve(self, rotor):
        """
        Find inflow angle and calculate forces for a single rotor given rotational speed, inflow velocity and radius.

        :rtype: tuple
        """

        rotor.precalc(self.scenario.twist)

        # Axial momentum (thrust)
        T = 0.0
        # Angular momentum
        Q = 0.0

        for sec in rotor.sections:

            if sec.radius < rotor.blade_radius:
                v = self.scenario.v_inf
            else:
                v = 0.0

            phi_lo = radians(5.0)  # degrees
            phi_hi = radians(30.0) # degrees

            # Find inflow angle
            if self.solver == 'brute':
                phi = self.brute_solve(sec, v, n=500, phi_lo=phi_lo, phi_hi=phi_hi)
            else:
                a_b = self._try_bracket(sec, v, phi_lo, phi_hi)
                if a_b != (None, None):
                    a, b = a_b
                    try:
                        phi = optimize.bisect(sec.func, a, b, args=(v, self.omega))
                    except Exception as e:
                        # Falhou bisect: usa brute como fallback
                        # print(e)
                        print('Bisect failed, switching to brute solver')
                        phi = self.brute_solve(sec, v, n=500, phi_lo=phi_lo, phi_hi=phi_hi)
                        phi = self._refine_with_secant(sec, v, phi, ddeg=1.0, iters=2)
                else:
                    # Não achou bracket → brute direto
                    phi = self.brute_solve(sec, v, n=500, phi_lo=phi_lo, phi_hi=phi_hi)
                    phi = self._refine_with_secant(sec, v, phi, ddeg=1.0, iters=2)
            
            dT, dQ = sec.forces(phi, v, self.omega, self.fluid)

            # Integrate
            T += dT
            Q += dQ
        
        # Power
        P = Q*self.omega  

        return T, Q, P
 
    def run(self, rotor:Rotor):
        """
        Runs the solver, i.e. finds the forces for each rotor.

        :return: Calculated thrust, torque, power and DataFrame with properties for all sections.
        :rtype: tuple
        """
        self.rotor = rotor
        self.T, self.Q, self.P = self.solve(self.rotor)
        
        return self.T, self.Q, self.P, self.rotor.sections_dataframe()

    def brute_solve(self, sec, v, n=3600, phi_lo=None, phi_hi=None):
        """ 
        Solve by a simple brute force procedure, iterating through all
        possible angles and selecting the one with lowest residual.

        :param Section sec: Section to solve for
        :param float v: Axial inflow velocity
        :param int n: Number of angles to test for, optional
        :param tuple phi_bounds: Tuple with lower and upper bounds for inflow angle, optional
        :return: Inflow angle with lowest residual
        :rtype: float
        """
        resid = np.zeros(n)

        if (phi_lo is not None) and (phi_hi is not None):
            phis = np.linspace(phi_lo, phi_hi, n)
        else:
            phis = np.linspace(-0.49*np.pi, 0.49*np.pi, n)

        for i,phi in enumerate(phis):
            res = sec.func(phi, v, self.omega)
            if not np.isnan(res):
                resid[i] = res
            else:
                resid[i] = 1e30
        i = np.argmin(abs(resid))
        return phis[i]
    
    def _try_bracket(self, sec, v, phi_lo, phi_hi):
        """
        Try to obtain (a,b) with sign change for sec.func inside [phi_lo, phi_hi].
        First tests the original edges; if it fails, expands around phi0 = atan2(v, omega*r).
        :param Section sec: Section to solve for
        :param float v: Axial inflow velocity
        :param float phi_lo: Lower bound for inflow angle
        :param float phi_hi: Upper bound for inflow angle
        :return: Tuple (a,b) with sign change, or (None,None) if it fails
        :rtype: tuple
        """
        def sgn(x):
            return np.sign(x) if np.isfinite(x) else 0.0

        # 1) testa bordas originais
        f_lo = sec.func(phi_lo, v, self.omega)
        f_hi = sec.func(phi_hi, v, self.omega)
        if sgn(f_lo) * sgn(f_hi) < 0:
            return phi_lo, phi_hi

        # 2) centro geométrico
        r = sec.radius
        if self.omega * r <= 1e-12:
            return (None, None)
        phi0 = np.arctan2(v, self.omega * r)

        # 3) expande simetricamente ao redor de phi0, mas SEM sair de [phi_lo, phi_hi]
        # passos de 1.5°, 3°, 6°, 12°...
        for k in range(8):
            d = np.deg2rad(1.5 * (2 ** k))
            a = np.clip(phi0 - d, phi_lo, phi_hi)
            b = np.clip(phi0 + d, phi_lo, phi_hi)
            fa = sec.func(a, v, self.omega)
            fb = sec.func(b, v, self.omega)
            if sgn(fa) * sgn(fb) < 0:
                return a, b

        return (None, None)
    
    def _refine_with_secant(self, sec, v, phi_star, ddeg=1.0, iters=2):
        """
        Refine a phi_star estimate with up to 'iters' secant steps,
        using samples at +/- ddeg degrees around it.
        """
        d = np.deg2rad(ddeg)
        a = phi_star - d
        b = phi_star + d
        fa = sec.func(a, v, self.omega)
        fb = sec.func(b, v, self.omega)

        if not (np.isfinite(fa) and np.isfinite(fb)):
            return phi_star
        if np.sign(fa) == np.sign(fb):
            return phi_star  # sem bracket local -> deixa como está

        for _ in range(iters):
            if abs(fb - fa) < 1e-14:
                break
            c = b - fb * (b - a) / (fb - fa)
            fc = sec.func(c, v, self.omega)
            a, fa = b, fb
            b, fb = c, fc
            if not np.isfinite(fb):
                break
        return b
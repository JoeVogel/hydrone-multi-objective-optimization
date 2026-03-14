# -*- coding: utf-8 -*-

"""
Module for storing rotor properties and calculation of induction factors and forces for the airfoil sections.
"""
import pandas as pd
import numpy as np
from configparser import NoOptionError
from math import radians, degrees, sqrt, cos, sin, atan2, atan, pi, acos, exp
from airfoil import load_airfoil


class Rotor: 
    """
    Holds rotor properties and a list of all airfoil sections.
    :param int n_blades: Number of blades
    :param float diameter: Rotor diameter
    :param float hub_radius: Radius of the hub
    :param list section_list: List of airfoil names
    :param list chord_list: List of chord lengths for each section
    :param list radius_list: List of radii for each section
    :param list pitch_list: List of pitch angles for each section
    """
    def __init__(self, n_blades, diameter, hub_radius, number_of_sections, foil_list, chord_list, pitch_list):
        self.n_blades = n_blades
        self.diameter = diameter
        self.hub_radius = hub_radius
        self.number_of_sections = number_of_sections

        _, centers = self.generate_rotor_section_points()

        s = foil_list
        c = chord_list
        r = centers
        
        S  = self.blade_radius - self.hub_radius
        dr = [S / self.number_of_sections] * self.number_of_sections
        
        self.alpha = pitch_list
        self.sections = []
        for i in range(self.number_of_sections): 
            sec = Section(load_airfoil(s[i]), float(r[i]), float(dr[i]), radians(self.alpha[i]), float(c[i]), self)
            self.sections.append(sec)
        
        self.hub_radius = hub_radius

        self.precalc(twist=0.0)

    def generate_rotor_section_points(self):
        """
        Generate rotor section radius list based on the number of sections, diameter and radius of the hub.

        :return: List of frontiers and centers for each section
        :rtype: tuple (list, list)
        """
        self.tip_radius = self.diameter / 2.0  # Tip radius of the propeller
        self.blade_radius = self.tip_radius      

        S = self.blade_radius - self.hub_radius  # usable span
        N = self.number_of_sections

        frontiers = [self.hub_radius + (i+1)*S/N for i in range(N)]
        centers   = [self.hub_radius + (i+0.5)*S/N for i in range(N)]
        return frontiers, centers

    def precalc(self, twist):
        """
        Calculation of properties before each solver run, to ensure all parameters are correct for parameter sweeps.
        This includes the blade radius and area, and applying the twist to each section.
        :return: None
        """
        self.blade_radius = 0.5*self.diameter
        self.area = pi*self.blade_radius**2

        # Apply twist
        for i,sec in enumerate(self.sections):
            sec.pitch = radians(self.alpha[i] + twist)

    def sections_dataframe(self):
        """
        Creates a pandas DataFrame with all calculated section properties.

        :return: DataFrame with section properties
        :rtype: pd.DataFrame
        """

        columns = ['foil_name','radius','chord','pitch','Cl','Cd','dT','dQ','F','a','ap','Re','AoA','U', 'Cp_min']
        data = {}
        for param in columns:
            array = [getattr(sec, param) for sec in self.sections]
            data[param] = array
        
        return pd.DataFrame(data)
 

class Section: 
    """
    Class for calculating induction factors and forces according to the BEM theory for a single airfoil section.

    :param Airfoil airfoil: Airfoil of the section
    :param float radius: Distance from center to mid of section
    :param float width: Width of section
    :param float pitch: Pitch angle in radians
    :param float chord: Chord length of section
    :param Rotor rotor: Rotor that section belongs to
    """
    def __init__(self, airfoil, radius, width, pitch, chord, rotor):
        self.airfoil = airfoil
        self.foil_name = airfoil.name
        self.radius = radius
        self.width = width
        self.pitch = pitch
        self.chord = chord
        self.rotor = rotor
        
        self.v = 0.0
        self.v_theta = 0.0
        self.v_rel = 0.0
        self.a=0.0
        self.ap=0.0
        self.Re = 0.0
        self.alpha = 0.0
        self.AoA = 0.0
        self.dT = 0.0
        self.dQ = 0.0
        self.F = 0.0
        self.Cl = 0.0
        self.Cd = 0.0
        self.U = 0.0
        self.Cp_min = 0.0

        self.precalc()
        
    def precalc(self):
        """
        Calculation of properties before each solver run, to ensure all parameters are correct for parameter sweeps.

        :return: None
        """
        self.sigma = self.rotor.n_blades*self.chord/(2*pi*self.radius) # local solidity

    def tip_loss(self, phi):
        """
        Prandtl tip + hub loss factor.
        Equations (scalar notation):
            F = (2/pi) * arccos( exp(-f) )
            f_tip = (B/2) * (R - r) / (r * |sin(phi)|)
            f_hub = (B/2) * (r - r_hub) / (r * |sin(phi)|)
        Returns F = F_tip * F_hub
        """
        B   = self.rotor.n_blades
        r   = self.radius
        R   = self.rotor.blade_radius
        rh  = self.rotor.hub_radius

        # Numerical guard rails
        eps_sin = 1e-12
        eps_r   = 1e-12

        # Use |sin(phi)| to avoid negative f when phi < 0
        s = abs(sin(phi))
        s = max(s, eps_sin)
        r_eff = max(r, eps_r)

        def prandtl(dr, r_local, s_local):
            """
            One-sided Prandtl factor for either tip (dr = R - r) or hub (dr = r - r_hub).
            """
            # If dr <= 0, no loss from that side
            if dr <= 0.0:
                return 1.0

            # f = (B/2) * dr / (r * |sin(phi)|)
            f = (B * dr) / (2.0 * r_local * s_local)

            # Avoid overflow/underflow in exp(-f) and keep acos argument valid
            # For f >= ~40, exp(-f) ~ 4e-18 -> arccos(0) ~ pi/2 => F ~ 1
            f = min(f, 60.0)  # safe upper cap
            z = np.exp(-f)
            z = float(np.clip(z, 0.0, 1.0))

            F_one_side = (2.0/pi) * acos(z)
            # Avoid exactly zero (would cause singularities downstream)
            return float(np.clip(F_one_side, 1e-4, 1.0))
        
        # Tip and hub components
        F_tip = prandtl(R - r_eff, r_eff, s)
        F_hub = prandtl(r_eff - rh, r_eff, s)

        F = F_tip * F_hub
        F = float(np.clip(F, 1e-4, 1.0))

        self.F = F
        return F
                    
    def airfoil_forces(self, phi, Re):
        """
        Force coefficients on an airfoil, decomposed in axial (C_T) and tangential (C_Q) directions:

        The coefficients are calculated using the airfoil lift (C_l) and drag (C_d) values from 
        the airfoil data tables, and the inflow angle (φ):
        
        .. math::
            C_T = C_l * cos(φ) - C_d * sin(φ)
            C_Q = C_l * sin(φ) + C_d * cos(φ)

        where:
            - φ (phi) is the inflow angle between the incoming airflow and the propeller’s plane of rotation,
             α (alpha) = C * (pitch - φ) is the effective angle of attack,
            - C_l and C_d are obtained from airfoil performance tables based on α.

        :param float phi: Inflow angle in radians
        :param float Re: Reynolds number
        :return: (C_T, C_Q) — axial and tangential force coefficients
        :rtype: tuple
        """

        alpha = self.pitch - phi
                
        Cl          = self.airfoil.Cl(alpha, Re)
        Cd          = self.airfoil.Cd(alpha, Re)
        self.Cp_min = self.airfoil.Cp_min(alpha, Re)
                
        CT = Cl*cos(phi) - Cd*sin(phi)
        CQ = Cl*sin(phi) + Cd*cos(phi)

        self.AoA = degrees(alpha)
        self.Cl = float(Cl)
        self.Cd = float(Cd)
        
        return CT, CQ
    
    def induction_factors(self, phi, Re=None):
        """
        Axial (a) and tangential (a') induction factors for propellers
            
        :param float phi: Inflow angle
        :param float Re: Reynolds number
        :return: Axial and tangential induction factors
        :rtype: tuple
        """
        
        F = self.tip_loss(phi)
        CT, CQ = self.airfoil_forces(phi, Re)

        # avoid singularities
        eps = 1e-6
        if F < eps:
            return 0.0, 0.0
        
        if abs(CT) < eps:
            CT = eps
        if abs(CQ) < eps:
            CQ = eps

        kappa  = 4.0 * F * sin(phi)**2          / (self.sigma * CT)
        kappap = 4.0 * F * sin(phi) * cos(phi)  / (self.sigma * CQ)

        # Linear induction factors for propellers
        a  = 1.0 / (kappa  - 1.0)
        ap = 1.0 / (kappap + 1.0)

        # Physical clamps
        if a < 0.0:  a = 0.0
        if a > 0.95: a = 0.95

        if ap < 0.0: ap = 0.0
        if ap > 1.0: ap = 1.0

        return a, ap
        
    def func(self, phi, v_inf, omega):
        """
        Residual function used in root-finding functions to find the inflow angle for the current section.

        .. math::
            \\frac{\\sin\\phi}{1+Ca} - \\frac{V_\\infty\\cos\\phi}{\\Omega r (1 - Ca\')} = 0\\\\

        We evaluate an algebraically equivalent residual without divisions to improve numerical robustness.

        :param float phi: Estimated inflow angle
        :param float v_inf: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :return: Residual
        :rtype: float
        """
        # Function to solve for a single blade element

        a, ap = self.induction_factors(phi)

        resid = sin(phi)/(1 + a) - v_inf*cos(phi)/(omega*self.radius*(1 - ap))

        self.a  = a
        self.ap = ap

        return resid
    
    def forces(self, phi, v_inf, omega, fluid, max_iter=4, tol=1e-5):
        """
        Compute dT and dQ for the section at given phi, including a short
        local fixed-point iteration to couple Re -> (Cl,Cd) -> (a,a').

        The definition of blade element theory is used,

        .. math::
            \\Delta T = \\sigma\\pi\\rho U^2C_T r\\Delta r \\\\
            \\Delta Q = \\sigma\\pi\\rho U^2C_Q r^2\\Delta r \\\\
            U = \\sqrt{v^2+v\'^2} \\\\
            v = (1 + Ca)V_\\infty \\\\
            v\' = (1 - Ca\')\\Omega R \\\\

        Note that this is equivalent to the momentum theory definition,

        .. math::
            \\Delta T = 4\\pi\\rho r V_\\infty^2(1 + Ca)aF\\Delta r \\\\
            \\Delta Q = 4\\pi\\rho r^3 V_\\infty\\Omega(1 + Ca)a\'F\\Delta r \\\\


        :param float phi: Inflow angle
        :param float v_inf: Axial inflow velocity
        :param float omega: Tangential rotational velocity
        :param Fluid fluid: Fluid 
        :param int max_iter: Maximum number of iterations for induction factor convergence
        :param float tol: Tolerance for induction factor convergence
        :return: Axial and tangential forces
        :rtype: tuple
        """

        r = self.radius
        rho = fluid.rho
        mu  = fluid.mu
        
        # First guess:
        a, ap = 0.0, 0.0

        Re_prev = None
        for _ in range(max_iter):
            # velocidades locais com a, a' correntes
            v  = (1 + a)  * v_inf
            vp = (1 - ap) * omega * r
            U  = sqrt(v**2 + vp**2)

            # Reynolds local
            Re = (rho * U * self.chord) / mu
            self.v_rel = U
            self.Re    = Re

            a_new, ap_new = self.induction_factors(phi, Re=Re)

            a, ap = a_new, ap_new

            # critério simples de convergência em Re (ou em a/ap)
            if (Re_prev is not None) and abs((Re - Re_prev)/(Re + 1e-16)) < tol:
                break

            Re_prev = Re

        # Store results
        self.a  = a
        self.ap = ap
        
        v = (1 + a)*v_inf
        vp = (1 - ap)*omega*r
        self.U = sqrt(v**2 + vp**2)   
        
        CT, CQ = self.airfoil_forces(phi, Re=self.Re)
            
        # From blade element theory
        self.dT = self.sigma*pi*rho*self.U**2*CT*r*self.width
        self.dQ = self.sigma*pi*rho*self.U**2*CQ*r**2*self.width

        # From momentum theory
        # dT = 4*pi*rho*r*self.v_inf**2*(1 + a)*a*F
        # dQ = 4*pi*rho*r**3*self.v_inf*(1 + a)*ap*self.omega*F
                
        return self.dT, self.dQ
        
    


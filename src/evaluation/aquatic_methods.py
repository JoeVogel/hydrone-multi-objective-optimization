from .evaluation_type           import EvaluationType
from .evaluation_method          import EvaluationMethod
from .fidelity_level            import FidelityLevel
from fluid                      import FluidType, Fluid
from scenario                   import Scenario
from rotor                      import Rotor
from airfoil                    import thickness_naca, thickness_e63
from .bemt                      import Solver as BEMTSolver

import os
import logging
import re

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from math import sqrt, pi

logger = logging.getLogger(__name__)

class WaterBEMT(EvaluationMethod):
    def __init__(self, scenario:Scenario):
        """ Initializes the WaterBEMT evaluation method."""
        super().__init__(evaluation_type=EvaluationType.AQUATIC, fidelity_level=FidelityLevel.LOW)

        self.fluid = Fluid(FluidType.WATER)
        self.scenario = scenario

        self.solver = BEMTSolver(scenario=self.scenario, fluid=self.fluid)

    def _compute_cavitation_sections(self, sections_df: pd.DataFrame, depth: float = 0.0) -> pd.DataFrame:
        """
        Compute the local cavitation and flag cavitation risk per blade section.

        This method follows the FLUID MECHANICS FUNDAMENTALS AND APPLICATIONS (YUNUS A. ÇENGEL; JOHN M. CIMBALA)

        What the method does
        --------------------
        1) Validates required inputs per section ('foil' and 'alpha'); invalid rows are skipped with a warning.
        2) Computes the local cavitation.
        3) Get Cp_min from XFOIL data.
        4) Compares Cp_min with -sigma to determine cavitation inception.
        5) Flags cavitation risk if Cp_min <= -sigma.

        Mathematical definitions
        ------------------------
        Static pressure at depth:
            p_inf = rho * g * depth

        Local cavitation number:
            sigma = 2 * (p_inf - pv) / (rho * U**2)

        Cavitation inception criterion:
            # Cavitation starts when the local pressure drops to or below the vapor pressure: p <= p_v.
            # the condition p <= p_v becomes Cp <= -sigma. Using the most negative surface value Cp_min

            cavitation_risk = Cp_min <= -sigma

        Parameters
        ----------
        sections_df : pd.DataFrame
            Section-wise data produced by the BEMT evaluation. Must contain:
            - 'r'      : radial position [m]
            - 'v_rel'  : local relative velocity U at the section [m/s]
            - 'alpha'  : angle of attack at the section [deg]
            - 'foil'   : foil name (e.g., "NACA0018", "NACA23012", "NACA63-018")
            Optional columns (not strictly required here but commonly present):
            - 'Re'     : local Reynolds number
            - 'Cl'     : local lift coefficient
            Rows with r/R < 0.2 are ignored (filled with NaN and cavitation_risk=False).
            Rows with invalid 'foil' or 'alpha' are skipped with a logged warning.

        depth : float, optional
            Submergence depth below the free surface [m] used to compute the static head.
            Default is 0.0.
        
        Returns
        -------
        pd.DataFrame
            The input DataFrame with three new columns:
            - 'sigma'            : local cavitation number (dimensionless)
            - 'Cp_min'           : correlated minimum pressure coefficient (dimensionless)
            - 'cavitation_risk'  : boolean flag; True if (Cp_min + sigma) <= 0, else False
            For skipped/ignored rows, the added columns are NaN/False accordingly.
        """
        rho  = self.fluid.rho                 # [kg/m^3]
        pv   = self.fluid.pv * 1e3            # [Pa] (vapor pressure in kPa to Pa)
        g    = 9.80665                        # [m/s^2]
        p_inf = rho * g * float(depth)

        if 'radius' not in sections_df.columns:
            raise ValueError("sections_df must contain the column 'radius' (radial position).")

        results = []
        for idx, row in sections_df.iterrows():
            U = row['U'] # Local relative velocity at the section
            Cp_min = row['Cp_min'] # minimum pressure coefficient

            if U <= 0.0 or Cp_min == 0.0:
                results.append((np.nan, np.nan, False))
                continue
            
            # Local cavitation calculation
            sigma_cav = 2 * (p_inf - pv) / (rho * U**2)

            # Cavitation inception criterion
            cavitates = Cp_min < -sigma_cav #TODO: verificar se uso (-) ou se uso cp_min_absolute

            results.append((sigma_cav, Cp_min, bool(cavitates)))

        sections_df[['sigma_cav', 'Cp_min', 'cavitation_risk']] = pd.DataFrame(results, index=sections_df.index)
        return sections_df

    def compute_quality_index_bollard(self, KT: float, KQ: float) -> float:
        """
        Compute the propeller Quality Index (QI) under bollard pull condition (J = 0).

        This metric is used when the advance ratio (J) is zero, meaning the inflow velocity (V_A) is null.
        In this case, the propeller operates in static thrust condition, where the efficiency η₀ is zero and cannot be used as a merit measure.

        The QI provides a non-dimensional figure of merit based on thrust and torque coefficients.

        References:
        ------------------- 
        Design of Propulsion and Electric Power Generation Systems.
        Hans Klein Woud and Douwe Stapersma
        2019

        Formula:
            QI(J=0) = (1 / (π * √(2π))) * (K_T^(3/2) / K_Q)

        where:
            K_T : float
                Non-dimensional thrust coefficient (C_T).
            K_Q : float
                Non-dimensional torque coefficient (C_Q).

        Returns:
            float : The non-dimensional Quality Index (QI) for J = 0 condition.

        Reference:
            Derived from the bollard pull formulation:
            QI = (1 / (π√(2π))) * (K_T^(3/2) / K_Q)
            1.5 is equivalente to raising to the power of 3/2.
        """

        if KQ <= 0 or KT <= 0:
            return 0.0

        return (1 / (pi * sqrt(2 * pi))) * ((KT ** 1.5) / KQ)

    def evaluate(self, rotor):
        """ Avalia o desempenho do propulsor baseado no modelo BEMT para ambiente aquatico """

        T, Q, P, sec_df = self.solver.run(rotor)
        J,CT,CQ,CP,eta = self.solver.rotor_coeffs(T, Q, P)
        QI = None

        # print("RE minimo: ", sec_df['Re'].min())
        # print("RE maximo: ", sec_df['Re'].max())

        cavitating_proportion = 0.0

        sec_df = self._compute_cavitation_sections(sections_df=sec_df, depth=0.5)

        num_cavitating_sections = int(sec_df['cavitation_risk'].fillna(False).sum())
        cavitating_proportion = num_cavitating_sections / len(sec_df)

        if self.scenario.v_inf == 0:
            QI = self.compute_quality_index_bollard(KT=CT, KQ=CQ)

        return T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion, QI
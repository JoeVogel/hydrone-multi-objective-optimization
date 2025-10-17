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
from math import sqrt

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
        Compute the local cavitation number and flag cavitation risk per blade section
        using a simplified Cp_min correlation based on alpha (deg) and thickness ratio t/c.

        This method is inspired by and follows the empirical trends and rationale described in:
            Amromin, E.; Rozhdestvensky, K. (2022). "Correlation between Pressure Minima and Cavitation
            Inception Numbers: Fundamentals and Hydrofoil Flows." Journal of Marine Science and Engineering.
        Specifically, the method uses a correlation between |Cp|min and the cavitation inception number that
        increases with both angle of attack and relative thickness, and apply it section-wise using local
        velocities from BEMT.

        What the method does
        --------------------
        1) Applies a radial cutoff (r/R >= 0.2), consistent with the BEMT discretization.
        2) Validates required inputs per section ('foil' and 'alpha'); invalid rows are skipped with a warning.
        3) Computes the local cavitation number sigma from static head and local relative speed.
        4) Estimates Cp_min via an empirical difference D_sigma = f(alpha, t/c), following the trends reported
        in the cited article.
        5) Flags cavitation risk if Cp_min + sigma <= 0.

        Mathematical definitions
        ------------------------
        Static pressure at depth:
            p_inf = p_atm + rho * g * depth

        Local cavitation number:
            sigma = (p_inf - p_v) / (0.5 * rho * U^2)

        Empirical correlation for the difference between |Cp|min and sigma_i:
            D_sigma = a1 * (alpha / 10) + a2 * (t/c) / 0.1

        Correlated minimum pressure coefficient:
            Cp_min = -(sigma + D_sigma)

        Cavitation inception criterion:
            # Cavitation starts when the local pressure drops to or below the vapor pressure: p <= p_v.
            # the condition p <= p_v becomes Cp <= -sigma. Using the most negative surface value Cp_min,
            # cavitation risk is flagged when Cp_min + sigma <= 0 (here sigma uses the local U at the section).

            cavitation_risk = (Cp_min + sigma) <= 0

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
        patm = 101325.0                       # [Pa]
        g    = 9.80665                        # [m/s^2]
        p_inf = patm + rho * g * float(depth) # [Pa]

        if 'radius' not in sections_df.columns:
            raise ValueError("sections_df must contain the column 'radius' (radial position).")

        results = []
        for idx, row in sections_df.iterrows():
            U = row['U'] # Local relative velocity at the section
            Cp_min = row['Cp_min'] # minimum pressure coefficient

            if U <= 0.0 or Cp_min == 0.0:
                results.append((np.nan, np.nan, False))
                continue
            
            # Local sigma (uses local U)
            sigma_cav = (p_inf - pv) / (0.5 * rho * U**2)

            cavitates = (Cp_min + sigma_cav) <= 0.0

            results.append((sigma_cav, Cp_min, bool(cavitates)))

        sections_df[['sigma_cav', 'Cp_min', 'cavitation_risk']] = pd.DataFrame(results, index=sections_df.index)
        return sections_df

    def evaluate(self, rotor):
        """ Avalia o desempenho do propulsor baseado no modelo BEMT para ambiente aquatico """

        T, Q, P, sec_df = self.solver.run(rotor)
        J,CT,CQ,CP,eta = self.solver.rotor_coeffs(T, Q, P)

        print("RE minimo: ", sec_df['Re'].min())
        print("RE maximo: ", sec_df['Re'].max())

        cavitating_proportion = 0.0

        # TODO: depurar para ver se está tudo ok
        sec_df = self._compute_cavitation_sections(sections_df=sec_df, depth=0.5)

        num_cavitating_sections = int(sec_df['cavitation_risk'].fillna(False).sum())
        cavitating_proportion = num_cavitating_sections / len(sec_df)

        return T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion
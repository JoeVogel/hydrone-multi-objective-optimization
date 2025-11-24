from .evaluation_method         import EvaluationMethod
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel
from fluid                      import FluidType, Fluid
from scenario                   import Scenario
from rotor                      import Rotor
from .bemt                      import Solver as BEMTSolver

from scipy.interpolate import interp1d
from math import sqrt

import numpy as np
import pandas as pd

import os
import logging

logger = logging.getLogger(__name__)


class AerialBEMT(EvaluationMethod):
    def __init__(self, scenario:Scenario):
        """ Initializes the AerialBEMT evaluation method."""
        super().__init__(evaluation_type=EvaluationType.AERIAL, fidelity_level=FidelityLevel.LOW)

        self.fluid = Fluid(FluidType.AIR)
        self.scenario = scenario

        self.solver = BEMTSolver(scenario=self.scenario, fluid=self.fluid)

    def _compute_FM(self, T, P, rotor:Rotor):
        """ 
        Compute Figure of Merit (FM) for hover condition. 
        
        Classic definition of FM for hover condition
        
        FM = Ideal Power / Actual Power = (T**(3/2)) / (P * sqrt(2 * rho * A))
        
        where
            T = Thrust [N]
            P = Power   [W]
            rho = fluid density [kg/m³] 
            A = pi * R**2 = pi * (D/2)**2 

        """

        R = rotor.diameter / 2.0
        A = np.pi * R**2
        rho = self.fluid.rho

        if P <= 0:
            return 0.0

        FM_raw = (T**(3/2)) / (P * sqrt(2 * rho * A))

        # Clamp FM between 0 and 1
        # if FM_raw > 1.0:
        #     logger.warning(
        #         "[FM WARNING] FM_raw=%.5f > 1.0 — ajustando para 1.0 "
        #         "(T=%.3f N, P=%.3f W, D=%.4f m, rho=%.3f, A=%.4f)",
        #         FM_raw, T, P, rotor.diameter, rho, A
        #     )
        #     return 1.0

        # if FM_raw < 0.0:
        #     logger.warning(
        #         "[FM WARNING] FM_raw=%.5f < 0 — ajustando para 0 "
        #         "(T=%.3f N, P=%.3f W, D=%.4f m)",
        #         FM_raw, T, P, rotor.diameter
        #     )
        #     return 0.0

        return FM_raw

    def evaluate(self, rotor):
        """ Avalia o desempenho do propulsor baseado no modelo BEMT para ambiente aéreo """

        FM = None # Figure of Merit

        T, Q, P, sec_df = self.solver.run(rotor)
        J,CT,CQ,CP,eta = self.solver.rotor_coeffs(T, Q, P)

        # print("RE minimo: ", sec_df['Re'].min())
        # print("RE maximo: ", sec_df['Re'].max())

        if self.scenario.v_inf == 0:
            if CP <= 0:
                FM = 0.0
            else:
                FM = self._compute_FM(T, P, rotor)

        return T, Q, P, J, CT, CQ, CP, eta, FM
        
        
    
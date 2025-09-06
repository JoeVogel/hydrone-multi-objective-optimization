from .evaluation_method         import EvaluationMethod
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel
from fluid                      import FluidType, Fluid
from scenario                   import Scenario
from rotor                      import Rotor
from .bemt                      import Solver as BEMTSolver

from scipy.interpolate import interp1d

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

    def evaluate(self, rotor):
        """ Avalia o desempenho do propulsor baseado no modelo BEMT para ambiente aéreo """

        T, Q, P, sec_df = self.solver.run(rotor)
        J,CT,CQ,CP,eta = self.solver.rotor_coeffs(T, Q, P)

        # print("RE minimo: ", sec_df['Re'][4])
        # print("RE maximo: ", sec_df['Re'].max())

        return T, Q, P, J, CT, CQ, CP, eta
        
        
    
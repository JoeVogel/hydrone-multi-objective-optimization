from .evaluation_method         import EvaluationMethod
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel

import numpy as np
import logging

logger = logging.getLogger(__name__)

class UIUCSurrogateModel(EvaluationMethod):
    def __init__(self):
        super().__init__(evaluation_type=EvaluationType.AERIAL, fidelity_level=FidelityLevel.LOW)

    def evaluate(self, param1, param2, param3):
        """
        Evaluate the individual using a surrogate model (specific to aerial evaluation).
        Return the fitness value.
        """
        # TODO:Example implementation (replace with actual model logic) 
        fitness = np.sum(-10 * np.exp(-0.2 * np.sqrt(param3**2 + param1**2)))
        
        logger.debug(f"PolynomialEvaluation Evaluation - Fitness: {fitness}")
        
        return fitness

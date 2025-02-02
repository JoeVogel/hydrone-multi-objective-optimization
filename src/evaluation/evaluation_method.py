from abc import ABC, abstractmethod
from enum import Enum, auto

class EvaluationType(Enum):
    AERIAL  = auto()
    AQUATIC = auto()
    
class FidelityLevel(Enum):
    LOW     = auto()
    MEDIUM  = auto()
    HIGH    = auto()

class EvaluationMethod(ABC):
    
    def __init__(self, evaluation_type: EvaluationType, fidelity_level: FidelityLevel):
        self.type = evaluation_type
        self.fidelity_level = fidelity_level
    
    @abstractmethod
    def evaluate(self, individual):
        """Evaluate the given individual and return a fitness value."""
        pass
        
from abc                        import ABC, abstractmethod    
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel

class EvaluationMethod(ABC):
    
    def __init__(self, evaluation_type: EvaluationType, fidelity_level: FidelityLevel):
        self.type = evaluation_type
        self.fidelity_level = fidelity_level
    
    @abstractmethod
    def evaluate(self, param1, param2, param3):
        
        """Evaluate the given individual and return a fitness value."""
        pass
        
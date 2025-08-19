from abc                        import ABC, abstractmethod    
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel
from rotor                      import Rotor

class EvaluationMethod(ABC):
    
    def __init__(self, evaluation_type: EvaluationType, fidelity_level: FidelityLevel):
        self.type = evaluation_type
        self.fidelity_level = fidelity_level
    
    @abstractmethod
    def evaluate(self, rotor:Rotor):
        
        """
        Evaluate the performance of the rotor based on the evaluation method.
        :param rotor: The rotor to be evaluated
        :return: Thrust, Torque, Power 
        """
        pass
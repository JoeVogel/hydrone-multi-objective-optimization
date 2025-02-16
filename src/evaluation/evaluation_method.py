from abc                        import ABC, abstractmethod    
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel

class EvaluationMethod(ABC):
    
    def __init__(self, evaluation_type: EvaluationType, fidelity_level: FidelityLevel):
        self.type = evaluation_type
        self.fidelity_level = fidelity_level
    
    @abstractmethod
    def evaluate(self, alpha, D, B, n=6000):
        
        """
        Calculates thrust, torque and efficinecy.
        
        Parameters:
            alpha (float): Propeller angle of attack 0-10 (degrees). 
            D (float): Propeller diameter (m).
            B (int): Blade number.
            n (int, optional): Propeller rotation in RPM (default: 6000)
        
        Returns:
            tuple: propeller thrust (N), propeller torque (nM), propeller efficiency
        
        """
        pass
        
from .evaluation_method         import EvaluationMethod
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel

from scipy.interpolate import interp1d

import numpy as np
import pandas as pd

import os
import logging

logger = logging.getLogger(__name__)

class UIUCSurrogateModel(EvaluationMethod):
    def __init__(self):
        super().__init__(evaluation_type=EvaluationType.AERIAL, fidelity_level=FidelityLevel.LOW)

    def evaluate(self, alpha, D, B, n=6000):
        """
        Evaluate the individual using a surrogate model (specific to aerial evaluation).
        Return the fitness value.
        """
        # TODO:Example implementation (replace with actual model logic) 
        fitness = np.sum(-10 * np.exp(-0.2 * np.sqrt(B**2 + J**2)))
        
        logger.debug(f"PolynomialEvaluation Evaluation - Fitness: {fitness}")
        
        return fitness

class AerialBEMT(EvaluationMethod):
    def __init__(self):
        super().__init__(evaluation_type=EvaluationType.AERIAL, fidelity_level=FidelityLevel.LOW)
        self.rho = 1.225  # Densidade do ar (kg/m³)
        self.mu = 1.81e-5  # Viscosidade dinâmica do ar (kg/m.s)
        self.N = 20  # Número de elementos ao longo do raio
        
        csv_path = os.path.join(os.path.dirname(__file__), "../../data/NACA0018_CL_and_CD_aerial_(Rogowski_Krolak_Bangga)_2021.csv")
        self.airfoil_data = pd.read_csv(csv_path)
        
        # Criar interpoladores para evitar problemas de falta de dados
        self.alpha_values = self.airfoil_data['Angle of Attack (deg)'].values
        self.cl_values = self.airfoil_data['C_L'].values
        self.cd_values = self.airfoil_data['C_D'].values

        self.cl_interp = interp1d(self.alpha_values, self.cl_values, kind='linear', fill_value="extrapolate")
        self.cd_interp = interp1d(self.alpha_values, self.cd_values, kind='linear', fill_value="extrapolate")

    def get_airfoil_coefficients(self, alpha):
        """ Retorna os coeficientes aerodinâmicos interpolados para um dado ângulo de ataque """
        self.C_L = float(self.cl_interp(alpha))
        self.C_D = float(self.cd_interp(alpha))

    def evaluate(self, alpha, D, B, n=6000):
        
        self.get_airfoil_coefficients(alpha)
        
        if self.C_L is None or self.C_D is None:
            raise ValueError(f"Coeficientes aerodinâmicos inválidos para ângulo de ataque {alpha}°.")
        
        R = D / 2  # Raio da hélice (m)
        dr = R / self.N  # Tamanho do elemento
        
        J = np.tan(np.radians(alpha))  # Calcula a razão de avanço a partir do ângulo de ataque
        V = J * n * D  # Velocidade axial (m/s)
        omega = 2 * np.pi * n / 60  # Velocidade angular (rad/s)
        
        dT_total = 0  # Empuxo total
        dQ_total = 0  # Torque total
        reynolds_numbers = []
        
        for i in range(1, self.N + 1):
            r = i * dr  # Raio do elemento
            
            V_a = V  # Velocidade axial (m/s)
            V_t = omega * r  # Velocidade tangencial
            V_rel = np.hypot(V_a, V_t)  # Velocidade relativa ao aerofólio
            
            phi = np.arctan2(V_a, V_t)  # Ângulo de fluxo
            
            c = 0.05 * D  # Corda da pá (~5% do diâmetro)
            sigma = (B * c) / (2 * np.pi * r)  # Razão de solidificação
            
            # Cálculo do número de Reynolds
            Re = (self.rho * V_rel * c) / self.mu
            reynolds_numbers.append(Re)
            
            dL = 0.5 * self.rho * V_rel**2 * c * self.C_L * dr  # Sustentação
            dD = 0.5 * self.rho * V_rel**2 * c * self.C_D * dr  # Arrasto
            
            dT = (dL * np.cos(phi)) - (dD * np.sin(phi))  # Empuxo
            dQ = r * ((dL * np.sin(phi)) + (dD * np.cos(phi)))  # Torque
            
            dT_total += dT * B
            dQ_total += dQ * B
        
        # Coeficientes adimensionais
        C_T = dT_total / (self.rho * (n * D)**2 * D**4)  # Coeficiente de empuxo
        C_Q = dQ_total / (self.rho * (n * D)**2 * D**5)  # Coeficiente de torque
        
        # Eficiência da hélice
        eta = (C_T * J) / (C_Q * 2 * np.pi) if C_Q != 0 else 0
        
        """
        TODO:
        
        Possíveis alterações:
            Se você quiser levar a classe para um nível ainda mais avançado, pode considerar:

            Implementar modelos de perda de ponta (Prandtl’s Tip Loss Model) para aumentar a precisão do modelo.
            Adaptar o método para calcular forças induzidas pelo fluxo (como indução axial e tangencial).
            Permitir que a geometria da pá varie ao longo do raio em vez de fixar a corda como 0.05 * D.
        """
        
        return dT_total, dQ_total, eta, reynolds_numbers
    
    
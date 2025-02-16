from .evaluation_method         import EvaluationMethod
from .evaluation_type           import EvaluationType
from .fidelity_level            import FidelityLevel

import os
import logging

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

class PolynomialEvaluation(EvaluationMethod):
    def __init__(self):
        super().__init__(evaluation_type=EvaluationType.AQUATIC, fidelity_level=FidelityLevel.HIGH)

    def evaluate(self, alpha, D, B, n=2000):
        """
        Evaluate the individual using polynomial calculations (specific to aquatic evaluation).
        Returns the fitness value.
        """
        # TODO:Example implementation (replace with actual model logic)
        a = 0.8
        b = 3
        fitness = np.sum(np.abs(alpha)**a + 5 * np.sin(alpha)**b)
        logger.debug(f"PolynomialEvaluation Evaluation - Fitness: {fitness}")
        
        return fitness

class WaterBEMT(EvaluationMethod):
    def __init__(self):
        super().__init__(evaluation_type=EvaluationType.AQUATIC, fidelity_level=FidelityLevel.LOW)
        self.rho = 997  # Densidade da água (kg/m³)
        self.mu = 8.9e-4  # Viscosidade dinâmica da água (kg/m.s)
        self.N = 20  # Número de elementos ao longo do raio
        
        csv_path = os.path.join(os.path.dirname(__file__), "../../data/NACA0018_CL_and_CD_water_(Oliveira)_2011_150000.csv")
        self.airfoil_data = pd.read_csv(csv_path)

        # Criar interpoladores para os coeficientes aerodinâmicos
        self.alpha_values = self.airfoil_data['Angle of Attack (deg)'].values
        self.cl_values = self.airfoil_data['C_L'].values
        self.cd_values = self.airfoil_data['C_D'].values

        self.cl_interp = interp1d(self.alpha_values, self.cl_values, kind='linear', fill_value="extrapolate")
        self.cd_interp = interp1d(self.alpha_values, self.cd_values, kind='linear', fill_value="extrapolate")

        # Parâmetros para cálculo de cavitação
        self.vapor_pressure = 2338  # Pressão de vapor da água (Pa) a 25°C
        self.p_atm = 101325  # Pressão atmosférica ao nível do mar (Pa)

    def get_airfoil_coefficients(self, alpha):
        """ Retorna os coeficientes aerodinâmicos interpolados para um dado ângulo de ataque """
        self.C_L = float(self.cl_interp(alpha))
        self.C_D = float(self.cd_interp(alpha))

    def evaluate(self, alpha, D, B, n=2000):
        """ Avalia o desempenho do propulsor baseado no modelo BEMT para ambiente aquático """
        
        depth=2
        
        self.get_airfoil_coefficients(alpha)

        if self.C_L is None or self.C_D is None:
            raise ValueError(f"Coeficientes aerodinâmicos inválidos para ângulo de ataque {alpha}°.")

        R = D / 2  # Raio da hélice (m)
        dr = R / self.N  # Tamanho do elemento
        
        J = np.tan(np.radians(alpha))  # Razão de avanço
        V = J * n * D  # Velocidade axial (m/s)
        omega = 2 * np.pi * n / 60  # Velocidade angular (rad/s)
        
        dT_total = 0  # Empuxo total
        dQ_total = 0  # Torque total
        cavitation_numbers = []  # Lista de números de cavitação
        reynolds_numbers = []  # Lista de números de Reynolds para cada seção

        for i in range(1, self.N + 1):
            r = i * dr  # Raio do elemento
            
            V_a = V  # Velocidade axial (m/s)
            V_t = omega * r  # Velocidade tangencial
            V_rel = np.hypot(V_a, V_t)  # Velocidade relativa ao aerofólio
            
            phi = np.arctan2(V_a, V_t)  # Ângulo de fluxo
            
            c = 0.05 * D  # Corda da pá (~5% do diâmetro)
            sigma = (B * c) / (2 * np.pi * r)  # Razão de solidificação
            
            # Cálculo do número de Reynolds local
            Re = (self.rho * V_rel * c) / self.mu
            reynolds_numbers.append(Re)
            
            dL = 0.5 * self.rho * V_rel**2 * c * self.C_L * dr  # Sustentação
            dD = 0.5 * self.rho * V_rel**2 * c * self.C_D * dr  # Arrasto
            
            dT = (dL * np.cos(phi)) - (dD * np.sin(phi))  # Empuxo
            dQ = r * ((dL * np.sin(phi)) + (dD * np.cos(phi)))  # Torque
            
            dT_total += dT * B
            dQ_total += dQ * B

            # Cálculo do número de cavitação (σ)
            """
            p_atm = Pressão atmosférica (101325 Pa)
            rho = Densidade da água (997 kg/m³)
            g = Gravidade (9.81 m/s²)
            depth = Profundidade (padrão: 2 m)
            vapor_pressure = Pressão de vapor da água (2338 Pa a 25°C)
            V_rel= Velocidade relativa no elemento
            """
            p_dynamic = 0.5 * self.rho * V_rel**2
            sigma_cav = (self.p_atm + self.rho * 9.81 * depth - self.vapor_pressure) / p_dynamic
            cavitation_numbers.append(sigma_cav)

        # Coeficientes adimensionais
        C_T = dT_total / (self.rho * (n * D)**2 * D**4)
        C_Q = dQ_total / (self.rho * (n * D)**2 * D**5)
        
        # Eficiência da hélice
        eta = (C_T * J) / (C_Q * 2 * np.pi) if C_Q != 0 else 0

        # Velocidade da ponta da pá
        # A velocidade da ponta da pá é crítica para evitar erosão por cavitação
        V_tip = omega * R

        return dT_total, dQ_total, eta, min(cavitation_numbers), V_tip, reynolds_numbers
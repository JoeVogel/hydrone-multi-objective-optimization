# nsgaii_test.py

import math
import random
from types import SimpleNamespace

import pandas as pd

from optimization.nsgaii import NSGAII
from evaluation.aerial_methods import AerialBEMT
from evaluation.aquatic_methods import WaterBEMT
from scenario import Scenario
from rotor import Rotor   # rotor está direto em src/


# ============================
# Configuração de teste (fake)
# ============================

def make_test_configs():
    """
    Gera um dicionário de configuração mínimo para testar o NSGA-II.
    """
    return {
        "problem": {
            "diameter": 0.3556,
            "number_of_sections": 30,
            "hub_radius": 0.01,
            "max_chord_global": 0.06,  # 6 cm, por exemplo
        },
        "nsga2": {
            "pop_size": 10,
            "generations": 3,
            "seed": 42,
            "mutation_rate": 0.1,
        },
        "variables": [
            {"name": "alpha", "type": "float", "min": 10.0, "max": 30.0},
            {"name": "n_blades", "type": "int", "min": 2, "max": 5},
            {
                "name": "airfoil_list",
                "type": "categorical",
                "choices": ["NACA0018", "E63", "NACA4412"],
            },
        ],
    }


# ====================
# Helpers de verificação
# ====================

def check_individual_geometry(nsga: NSGAII, ind):
    """
    Verificações de sanidade da geometria de um indivíduo:
    - tamanhos das listas
    - restrição no hub
    - chord mínimo físico
    - suavidade radial de pitch e chord
    - foils válidos
    """
    n = nsga.number_of_sections

    assert len(ind.pitch_list) == n
    assert len(ind.chord_list) == n
    assert len(ind.foil_list) == n

    # chord na primeira seção não pode ser maior que o diâmetro do hub
    assert ind.chord_list[0] <= nsga.hub_diameter

    # chord mínimo físico (mesmo critério usado na mutação)
    min_chord = 0.015 * nsga.diameter
    assert all(c >= min_chord for c in ind.chord_list), "Chord menor que min_chord encontrado"

    # suavidade global (valores alinhados com _adjust_phisical_constraints)
    max_delta_pitch = (nsga.max_alpha - nsga.min_alpha) * 0.2
    max_delta_chord = 0.01

    eps = 1e-9  # numeric precision (floating point)

    for i in range(1, n):
        dp = abs(ind.pitch_list[i] - ind.pitch_list[i - 1])
        dc = abs(ind.chord_list[i] - ind.chord_list[i - 1])
        assert dp <= max_delta_pitch + eps, f"Delta pitch grande em seção {i}: {dp}"
        assert dc <= max_delta_chord + eps, f"Delta chord grande em seção {i}: {dc}"

    # todos os foils devem estar dentro das opções definidas
    for f in ind.foil_list:
        assert f in nsga.foil_options

def sample_test_crossover_mutation(nsga: NSGAII, n_samples=20):
    """
    Gera alguns filhos a partir de pais da população inicial,
    passando por crossover + mutação, e verifica se a geometria
    resultante continua física.
    """
    pop = nsga._initialize_population()

    for _ in range(n_samples):
        parent1 = random.choice(pop)
        parent2 = random.choice(pop)

        child = nsga._crossover(parent1, parent2)

        check_individual_geometry(nsga, child)

        nsga._mutate(child)

        check_individual_geometry(nsga, child)


# ============================
# Testes de integração (pytest)
# ============================

def test_nsga_run_small_config():
    """
    Integração básica:
    - Instancia NSGAII com métodos de avaliação fake.
    - Roda poucas gerações.
    - Verifica se terminou sem exceções, se gerou frentes,
      e se a geometria do primeiro indivíduo da primeira frente é válida.
    """
    configs = make_test_configs()
    aerial_eval = AerialBEMT(scenario=Scenario(rpm=4000.0, v_inf=0.0))
    aquatic_eval = WaterBEMT(scenario=Scenario(rpm=400.0, v_inf=0.0))

    nsga = NSGAII(configs, aerial_eval, aquatic_eval)

    fronts = nsga.run()

    # Deve ter pelo menos uma frente
    assert len(fronts) > 0

    # Número total de indivíduos nas frentes deve ser igual ao pop_size
    total_in_fronts = sum(len(f) for f in fronts)
    assert total_in_fronts == nsga.population_size

    # Checa o primeiro indivíduo da primeira frente
    best = fronts[0][0]
    check_individual_geometry(nsga, best)

    # Fitness deveria ter sido calculada
    assert best.aerial_fitness is not None
    assert best.aquatic_fitness is not None

    # CSV de avaliações deve existir e ter linhas
    df = pd.read_csv(nsga.eval_csv_path)
    assert not df.empty
    assert "generation" in df.columns
    assert "aerial_fitness" in df.columns
    assert "aquatic_fitness" in df.columns


def test_crossover_and_mutation_sampling():
    """
    Integração de operadores:
    - Usa um NSGAII com config de teste.
    - Gera vários filhos via crossover + mutação.
    - Verifica por amostragem se todos os filhos respeitam
      as restrições físicas básicas.
    """
    configs = make_test_configs()
    aerial_eval = AerialBEMT(scenario=Scenario(rpm=4000.0, v_inf=0.0))
    aquatic_eval = WaterBEMT(scenario=Scenario(rpm=400.0, v_inf=0.0))

    nsga = NSGAII(configs, aerial_eval, aquatic_eval)

    # testa 50 filhos, por exemplo
    sample_test_crossover_mutation(nsga, n_samples=50)

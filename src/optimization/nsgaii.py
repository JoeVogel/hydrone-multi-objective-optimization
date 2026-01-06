import os
import random
import logging
import csv
import math
import copy

import pandas as pd

from datetime import datetime
from pathlib import Path

from evaluation.evaluation_method import EvaluationMethod
from .individual                  import Individual
from rotor                        import Rotor

logger = logging.getLogger(__name__)

"""
The NSGA-II algorithm is a popular genetic algorithm for solving multi-objective optimization problems. 
It was first proposed by Deb et al. in 2002 as an improvement over the original Non-dominated Sorting Genetic Algorithm (NSGA).
https://ieeexplore.ieee.org/abstract/document/996017?casa_token=dDGNqrFAX2YAAAAA:zLPH-VzJoi06EqWav3xZuAB8pgTXSV7SsAm7sLSv9yU4g_-VStrzhtsJjnDwe6hTkMeuXAf1nA
"""

# TODO: avaliar implementação exemplo:
# https://github.com/smkalami/nsga2-in-python/blob/main/nsga2.py

class NSGAII:
    def __init__(self, aerial_evaluation_method: EvaluationMethod, aquatic_evaluation_method: EvaluationMethod, problem_configuration, motor_data, nsga_configuration):
        
        # TODO: adicionar checks nas configurações
        
        self.aerial_evaluation_method   = aerial_evaluation_method
        self.aquatic_evaluation_method  = aquatic_evaluation_method
        
        self.diameter           = problem_configuration["diameter"]
        self.number_of_sections = problem_configuration["number_of_sections"]
        self.hub_radius         = problem_configuration["hub_radius"]
        self.max_chord_global   = problem_configuration["max_chord_global"]
        self.min_alpha          = problem_configuration["min_alpha"] 
        self.max_alpha          = problem_configuration["max_alpha"]
        self.min_blade_number   = problem_configuration["min_blade_number"]
        self.max_blade_number   = problem_configuration["max_blade_number"]
        self.foil_options       = problem_configuration["foil_options"]

        self.population_size        = nsga_configuration["population_size"]
        self.maximum_generations    = nsga_configuration["maximum_generations"]
        self.seed                   = nsga_configuration["seed"]
        self.elitism_fraction       = nsga_configuration["elitism_fraction"]
        self.mutation_rate          = nsga_configuration["mutation_rate"]

        self.aerial_Q_max   = motor_data["aerial_Q_max"]
        self.aquatic_Q_max  = motor_data["aquatic_Q_max"]

        self.hub_diameter = self.hub_radius * 2 

        self._init_run_outputs()

    def _init_run_outputs(self):
        """Creates data/results/<datetime> and initializes evaluations.csv with header."""
        base_dir = Path(__file__).resolve().parent
        self.run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = (base_dir / "../results" / self.run_ts).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.eval_csv_path   = self.run_dir / "evaluations.csv"
        self.front_csv_path  = self.run_dir / "pareto_front.csv"

        # Fixed CSV fields (lists will be serialized as strings)
        self._eval_fields = [
            # identification
            "generation", "pop_index",
            # decision variables / geometry
            "D", "B", "hub_radius", "number_of_sections",
            "foil_list", "chord_list", "pitch_list",
            # Scenario (for traceability)
            "aerial_rpm", "aerial_v_inf", "aquatic_rpm", "aquatic_v_inf",
            # aerial metrics
            "aerial_T","aerial_Q","aerial_P","aerial_J","aerial_CT","aerial_CQ","aerial_CP","aerial_eta", "FM", "aerial_fitness", "aerial_Q_penalty",
            # water metrics
            "aquatic_T","aquatic_Q","aquatic_P","aquatic_J","aquatic_CT","aquatic_CQ","aquatic_CP","aquatic_eta", "cavitating_proportion", "QI","aquatic_fitness", "cavitation_penalty", "aquatic_Q_penalty",
        ]

        with self.eval_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._eval_fields)
            writer.writeheader()

        logger.info(f"[NSGA-II] Result dir: {self.run_dir}")

    @staticmethod
    def _safe_real(x):
        """Check if value is a valid real number. If not return 0.0"""
        if isinstance(x, complex):
            return 0.0
        if x is None or not math.isfinite(float(x)):
            return 0.0
        return float(x)

    @staticmethod
    def _serialize_list(value):
        """Converts lists/tuples to 'v1;v2;v3' string. Keeps '' for None."""
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ";".join(str(v) for v in value)
        return str(value)

    def _append_eval_row(self, generation, pop_index, individual, aerial, aquatic, target_path=None):
        """Appends a row to evaluations.csv or another specified CSV file."""
        row = {
            "generation": generation,
            "pop_index": pop_index,

            "D": getattr(individual, "D", ""),
            "B": getattr(individual, "B", ""),
            "hub_radius": getattr(individual, "hub_radius", ""),
            "number_of_sections": getattr(individual, "number_of_sections", ""),

            "foil_list": self._serialize_list(getattr(individual, "foil_list", None)),
            "chord_list": self._serialize_list(getattr(individual, "chord_list", None)),
            "pitch_list": self._serialize_list(getattr(individual, "pitch_list", None)),

            "aerial_rpm": getattr(self.aerial_evaluation_method.scenario, "rpm", ""),
            "aerial_v_inf": getattr(self.aerial_evaluation_method.scenario, "v_inf", ""),
            "aquatic_rpm": getattr(self.aquatic_evaluation_method.scenario, "rpm", ""),
            "aquatic_v_inf": getattr(self.aquatic_evaluation_method.scenario, "v_inf", ""),

            "aerial_T":   "" if aerial  is None else aerial.get("T",""),
            "aerial_Q":   "" if aerial  is None else aerial.get("Q",""),
            "aerial_P":   "" if aerial  is None else aerial.get("P",""),
            "aerial_J":   "" if aerial  is None else aerial.get("J",""),
            "aerial_CT":  "" if aerial  is None else aerial.get("CT",""),
            "aerial_CQ":  "" if aerial  is None else aerial.get("CQ",""),
            "aerial_CP":  "" if aerial  is None else aerial.get("CP",""),
            "aerial_eta": "" if aerial  is None else aerial.get("eta",""),
            "FM": "" if aerial  is None else aerial.get("FM",""),
            "aerial_Q_penalty": "" if aerial  is None else aerial.get("Q_penalty",""),
            "aerial_fitness": getattr(individual, "aerial_fitness", ""),

            "aquatic_T":   "" if aquatic is None else aquatic.get("T",""),
            "aquatic_Q":   "" if aquatic is None else aquatic.get("Q",""),
            "aquatic_P":   "" if aquatic is None else aquatic.get("P",""),
            "aquatic_J":   "" if aquatic is None else aquatic.get("J",""),
            "aquatic_CT":  "" if aquatic is None else aquatic.get("CT",""),
            "aquatic_CQ":  "" if aquatic is None else aquatic.get("CQ",""),
            "aquatic_CP":  "" if aquatic is None else aquatic.get("CP",""),
            "aquatic_eta": "" if aquatic is None else aquatic.get("eta",""),
            "cavitating_proportion": "" if aquatic is None else aquatic.get("cavitating_proportion",""),
            "aquatic_Q_penalty": "" if aquatic is None else aquatic.get("Q_penalty",""),
            "cavitation_penalty": "" if aquatic is None else aquatic.get("cavitation_penalty",""),
            "QI": "" if aquatic is None else aquatic.get("QI",""),
            "aquatic_fitness": getattr(individual, "aquatic_fitness", ""),
        }

        path = self.eval_csv_path if target_path is None else Path(target_path)

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._eval_fields)
            writer.writerow(row)

    def _write_pareto_front_csv(self, front):
        """Writes the first Pareto front to pareto_front.csv, recalculating metrics for traceability."""
        with self.front_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._eval_fields)
            writer.writeheader()
        
        for i, ind in enumerate(front):
            # Reconstrói o rotor a partir do indivíduo
            rotor = Rotor(
                n_blades=ind.B,
                diameter=ind.D,
                hub_radius=ind.hub_radius,
                number_of_sections=ind.number_of_sections,
                foil_list=ind.foil_list,
                chord_list=ind.chord_list,
                pitch_list=ind.pitch_list
            )

            a = w = None

            aT, aQ, aP, aJ, aCT, aCQ, aCP, aEta, FM = self.aerial_evaluation_method.evaluate(rotor)
            aQ_penalty = self._torque_penalty(aQ, self.aerial_Q_max)

            a = {"T": aT, "Q": aQ, "P": aP, "J": aJ, "CT": aCT, "CQ": aCQ, "CP": aCP, "eta": aEta, "FM": FM, "Q_penalty": aQ_penalty}

            wT, wQ, wP, wJ, wCT, wCQ, wCP, wEta, cavitating_proportion, QI = self.aquatic_evaluation_method.evaluate(rotor)
            cavitation_penalty = self._cavitation_penalty(cavitating_proportion)
            wQ_penalty = self._torque_penalty(wQ, self.aquatic_Q_max)

            w = {"T": wT, "Q": wQ, "P": wP, "J": wJ, "CT": wCT, "CQ": wCQ, "CP": wCP, "eta": wEta, "cavitating_proportion": cavitating_proportion, "QI": QI, "cavitation_penalty": cavitation_penalty, "Q_penalty": wQ_penalty}

            self._append_eval_row(
                generation="",  # último gen
                pop_index="",
                individual=ind,
                aerial=a,
                aquatic=w,
                target_path=self.front_csv_path
            )

    def _initialize_population(self):
        """
        Creates a highly diverse initial population using multiple
        blade profile archetypes.
        Each archetype generates chord/pitch/foil with different
        radial patterns to avoid all individuals falling in the same front.
        """

        population = []
        random.seed(self.seed)
        n = self.number_of_sections
        min_chord = 0.015 * self.diameter

        # -------- Archetype generators -------- #

        def profile_linear():
            """Linear chord + linear pitch + foil in blocks"""
            # chord
            root = 0.6 * self.hub_diameter
            tip  = random.uniform(0.15*root, 0.6*root)
            chord = [root + (tip - root)*(i/(n-1)) for i in range(n)]

            # pitch
            root_pitch = random.uniform(self.min_alpha, self.max_alpha)
            tip_pitch  = random.uniform(self.min_alpha, root_pitch)
            pitch = [root_pitch + (tip_pitch-root_pitch)*(i/(n-1)) for i in range(n)]

            # foils in blocks
            foil = []
            block = random.randint(3,8)
            for i in range(n):
                if i % block == 0:
                    f = random.choice(self.foil_options)
                foil.append(f)

            return chord, pitch, foil

        def profile_exponential():
            """Exponential chord decay + random local twist"""
            r = [i/(n-1) for i in range(n)]

            root = 0.6 * self.hub_diameter
            chord = [max(min_chord, root * (0.25 + 0.75 * (1 - ri)**2)) for ri in r]

            base = random.uniform(self.min_alpha, self.max_alpha)
            pitch = [max(self.min_alpha,
                        min(self.max_alpha, base + random.uniform(-3,3)*ri))
                    for ri in r]

            foil = [random.choice(self.foil_options) for _ in range(n)]
            return chord, pitch, foil

        def profile_s_curve():
            """S-curve chord + smooth twist pitch"""
            r = [i/(n-1) for i in range(n)]

            root = 0.6 * self.hub_diameter
            tip  = random.uniform(0.15, 0.25) * root
            chord = [root + (tip-root)*(3*r*r - 2*r*r*r) for r in r]
            chord = [max(min_chord, c) for c in chord]

            root_pitch = random.uniform(self.min_alpha, self.max_alpha)
            tip_pitch = random.uniform(self.min_alpha, root_pitch)
            pitch = [root_pitch + (tip_pitch-root_pitch)*(3*r*r - 2*r*r*r) for r in r]

            foil = []
            last = random.choice(self.foil_options)
            for i in range(n):
                if random.random() < 0.2:
                    last = random.choice(self.foil_options)
                foil.append(last)

            return chord, pitch, foil

        def profile_random_chaos():
            """High randomness but later smoothed — exploration archetype"""
            chord = [max(min_chord,
                        random.uniform(0.3*self.hub_diameter, self.hub_diameter))
                    for _ in range(n)]

            pitch = [random.uniform(self.min_alpha, self.max_alpha)
                    for _ in range(n)]

            foil = [random.choice(self.foil_options) for _ in range(n)]
            return chord, pitch, foil

        archetypes = [
            profile_linear,
            profile_exponential,
            profile_s_curve,
            profile_random_chaos,
        ]

        # -------- Generate individuals -------- #

        for _ in range(self.population_size):

            generator = random.choice(archetypes)
            chord, pitch, foil = generator()

            # Apply smoothing constraints
            pitch, chord = self._adjust_phisical_constraints(pitch, chord)

            individual = Individual(
                D=self.diameter,
                B=random.randint(self.min_blade_number, self.max_blade_number),
                pitch_list=pitch,
                chord_list=chord,
                foil_list=foil,
                hub_radius=self.hub_radius,
                number_of_sections=self.number_of_sections
            )

            population.append(individual)

        return population

    def _fast_non_dominated_sort(self, population):
        """
        Performs non-dominated sorting on the population.
        Returns a list of fronts.
        """
        fronts = [[]]
        for p in population:
            p.dominated_solutions = []
            p.domination_count = 0
            p.rank = None
            p.crowding_distance = 0.0

            for q in population:
                if self._dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self._dominates(q, p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            fronts.append(next_front)
            i += 1

        return fronts[:-1]

    def _crowding_distance(self, front):
        """
        Calculates the crowding distance for each individual in a front.
        """
        if not front:
            return

        for individual in front:
            individual.crowding_distance = 0

        for m in range(2):  # Two objectives: aerial and aquatic fitness
            # Ordena pelo objetivo atual
            if m == 0:
                front.sort(key=lambda x: x.aerial_fitness)
            else:
                front.sort(key=lambda x: x.aquatic_fitness)
            
            # Define distância infinita para as extremidades    
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # Calcula a distância para os indivíduos internos
            min_value = front[0].aerial_fitness if m == 0 else front[0].aquatic_fitness
            max_value = front[-1].aerial_fitness if m == 0 else front[-1].aquatic_fitness

            if max_value - min_value == 0:
                continue  # Evita divisão por zero

            for i in range(1, len(front) - 1):
                next_fitness = front[i + 1].aerial_fitness if m == 0 else front[i + 1].aquatic_fitness
                prev_fitness = front[i - 1].aerial_fitness if m == 0 else front[i - 1].aquatic_fitness

                front[i].crowding_distance += (next_fitness - prev_fitness) / (max_value - min_value)

    def _dominates(self, individual1: Individual, individual2: Individual):
        """
        Checks if individual1 dominates individual2.
        An individual dominates another if:
        - It is better (higher value) in at least one objective
        - It is not worse in any objective
        """
        
        better_in_one = False

        # Comparação para cada objetivo (fitness aéreo e fitness aquático)
        if individual1.aerial_fitness > individual2.aerial_fitness:  # Agora queremos valores maiores
            better_in_one = True
        elif individual1.aerial_fitness < individual2.aerial_fitness:
            return False  # Se piora em algum objetivo, não domina

        if individual1.aquatic_fitness > individual2.aquatic_fitness:  # Agora queremos valores maiores
            better_in_one = True
        elif individual1.aquatic_fitness < individual2.aquatic_fitness:
            return False  # Se piora em algum objetivo, não domina

        return better_in_one

    def _create_next_generation(self, population, fronts):
        """
        Creates the next generation using tournament selection, crossover, and mutation.
        Only a fraction of the population is filled by elitism (copying from fronts), and the rest is filled by offspring.
        """
        next_generation = []
        i = 0

        elite_max = max(1, int(self.population_size * self.elitism_fraction))

        # 1) Add entire fronts until we reach the elite limit
        while i < len(fronts) and len(next_generation) + len(fronts[i]) <= elite_max:
            next_generation.extend(fronts[i])
            i += 1

        # 2) If next front would exceed elite limit, fill remaining elite spots based on crowding distance
        remaining_elite_spots = elite_max - len(next_generation)
        if remaining_elite_spots > 0 and i < len(fronts):
            fronts[i].sort(key=lambda x: -x.crowding_distance)  # Ordem decrescente de crowding distance
            next_generation.extend(fronts[i][:remaining_elite_spots])  # Preenche com os melhores

        # 3) Complete the rest of the population with offspring (crossover + mutation)
        while len(next_generation) < self.population_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            child = self._crossover(parent1, parent2)
            self._mutate(child)

            next_generation.append(child)

        return next_generation

    def _tournament_selection(self, population, tournament_size=2):
        """
        Perform tournament selection to choose one individual from the population.
        """
        # Randomly select individuals for the tournament
        tournament = random.sample(population, tournament_size)
        
        # Sort by rank first, then by crowding distance (higher is better)
        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        # Return the best individual from the tournament
        return tournament[0]

    def _smooth_profile(self, values, max_delta):
        """
        Limits the jump between neighboring sections to |values[i] - values[i-1]| <= max_delta.
        Applies a progressive "clamping".
        """
        if not values:
            return values

        smoothed = list(values)
        for i in range(1, len(smoothed)):
            delta = smoothed[i] - smoothed[i - 1]
            if abs(delta) > max_delta:
                smoothed[i] = smoothed[i - 1] + max_delta * math.copysign(1, delta)
        return smoothed

    def _enforce_chord_constraints(self, chord_list):
        """
        First section chord cannot be larger than hub diameter.
        """
        if not chord_list:
            return chord_list

        chord_list = list(chord_list)
        if chord_list[0] > self.hub_diameter * 0.6:
            chord_list[0] = self.hub_diameter * 0.6
        return chord_list

    def _adjust_phisical_constraints(self, pitch_list, chord_list):
        """
        Adjusts pitch and chord lists to respect physical constraints.
        """
        # TODO: testar e setar valores adequados
        max_delta_pitch = (self.max_alpha - self.min_alpha) * 0.2  # Exemple: 20% of range
        max_delta_chord = 0.01 # Exemple: 1 cm

        # Respect hub diameter constraint
        chord_list = self._enforce_chord_constraints(chord_list)

        pitch_list = self._smooth_profile(pitch_list, max_delta_pitch)
        chord_list = self._smooth_profile(chord_list, max_delta_chord)

        # Second pass to ensure constraints
        chord_list = self._enforce_chord_constraints(chord_list)

        return pitch_list, chord_list

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two parents to produce a child.
        """

        n = self.number_of_sections
        cut = random.randint(1, n - 1)

        child_pitch = list(parent1.pitch_list[:cut]) + list(parent2.pitch_list[cut:])
        child_chord = list(parent1.chord_list[:cut]) + list(parent2.chord_list[cut:])

        child_foil  = list(parent1.foil_list[:cut])  + list(parent2.foil_list[cut:])

        # --- Cut region pitch and chord smoothing ---
        # Pitch and chord in sections cut-1 and cut can be interpolated
        if 1 <= cut < n:
            # Pitch
            p_left  = parent1.pitch_list[cut - 1]
            p_right = parent2.pitch_list[cut]
            interp_pitch = 0.5 * (p_left + p_right)
            child_pitch[cut - 1] = interp_pitch
            child_pitch[cut]     = interp_pitch

            # Chord
            c_left  = parent1.chord_list[cut - 1]
            c_right = parent2.chord_list[cut]
            interp_chord = 0.5 * (c_left + c_right)
            child_chord[cut - 1] = interp_chord
            child_chord[cut]     = interp_chord

        # --- Post-smoothing to avoid jagged profiles ---
        child_pitch, child_chord = self._adjust_phisical_constraints(child_pitch, child_chord)

        child_B = random.choice([parent1.B, parent2.B])

        # Create child individual
        child = Individual(
            D=self.diameter,
            B=child_B,
            pitch_list=child_pitch,
            chord_list=child_chord,
            foil_list=child_foil,
            hub_radius=self.hub_radius,
            number_of_sections=self.number_of_sections
        )

        return child

    def _choose_block_indices(self, n: int, block_fraction: float):
        """
        Chooses a contiguous block of indices based on selected fraction.
        """
        if n <= 0:
            return []

        block_size = max(1, int(round(block_fraction * n)))
        block_size = min(block_size, n)

        start = random.randint(0, n - block_size)
        return list(range(start, start + block_size))

    def _mutate(self, individual):
        """
        Perform mutation on an individual's parameters.
        """

        n = self.number_of_sections

        pitch_list = list(individual.pitch_list)
        chord_list = list(individual.chord_list)
        foil_list  = list(individual.foil_list)

        n_mutations = max(1, int(self.mutation_rate * n)) # at least one mutation
        n_mutations = min(n_mutations, n) # cannot exceed number of sections

        # Mutate number of blades
        if random.random() < self.mutation_rate:
            # Mutate number of blades
            individual.B = random.randint(self.min_blade_number, self.max_blade_number)

        # Mutate pitch
        sigma_pitch = 1.0  # Standard deviation for pitch mutation (in degrees)

        pitch_block_idx = self._choose_block_indices(n, self.mutation_rate)

        # A single delta for the entire block
        delta_pitch = random.gauss(0.0, sigma_pitch)

        for i in pitch_block_idx:
                new_pitch = pitch_list[i] + delta_pitch
                new_pitch = max(self.min_alpha, min(self.max_alpha, new_pitch))
                pitch_list[i] = new_pitch
                 
        # Mutate chord
        chord_block_idx = self._choose_block_indices(n, self.mutation_rate)

        # A single scaling factor for the entire block
        factor_chord = 1.0 + random.uniform(-0.1, 0.1)  # ±10%

        # Minimum chord to avoid non-physical values
        min_chord = 0.015 * self.diameter # 1.5% of diameter

        for i in chord_block_idx:
                new_chord = chord_list[i] * factor_chord

                if new_chord < min_chord:
                    new_chord = min_chord

                if self.max_chord_global is not None:
                    new_chord = min(new_chord, self.max_chord_global)

                chord_list[i] = new_chord

        # Mutate foil
        foil_block_idx = self._choose_block_indices(n, self.mutation_rate)

        new_foil = random.choice(self.foil_options)

        for i in foil_block_idx:
            foil_list[i] = new_foil

        # Adjust to respect physical constraints
        pitch_list, chord_list = self._adjust_phisical_constraints(pitch_list, chord_list)

        # Update individual
        individual.pitch_list = pitch_list
        individual.chord_list = chord_list
        individual.foil_list  = foil_list

        individual.aerial_fitness  = None
        individual.aquatic_fitness = None
        individual.cavitating_proportion = None

    def _cavitation_penalty(self, cav):
        """
        Calculates a penalty based on the cavitating proportion.
        """
        
        #linear penalty: 0% cavitation = 0 penalty, 100% cavitation = 100% penalty
        penalty = max(0.0, min(1.0, cav))  # ensure cav is between 0 and 1
        return penalty

    def _torque_penalty(self, T, T_max):
        """
        Returns a penalty based on how much the required torque exceeds T_max.

        Output is clamped between 0 and 1.
        """

        if T <= T_max:
            return 0.0

        penalty = (T - T_max) / T

        penalty = max(0.0, min(1.0, penalty))

        return penalty

    def _compute_aerial_fitness(self, raw_result, aQ_penalty):
        """
        Computes aerial fitness based on efficiency and torque penalty.
        """
        fitness = raw_result * (1.0 - aQ_penalty)
        return fitness
    
    def _compute_aquatic_fitness(self, raw_result, cavitation_penalty, wQ_penalty):
        """
        Computes aquatic fitness based on efficiency, cavitation penalty, and torque penalty.
        """
        combined_factor = (1.0 - cavitation_penalty) * (1.0 - wQ_penalty)
        combined_factor = max(0.0, min(1.0, combined_factor))

        fitness = raw_result * combined_factor
        return fitness

    def run(self):
        """
        Executes the NSGA-II optimization process and returns Pareto fronts.
        """
        population = self._initialize_population()
        for generation in range(1, self.maximum_generations + 1):
            logger.info(f"Generation {generation}/{self.maximum_generations}")

            # Evaluate population
            for idx, individual in enumerate(population):

                rotor = Rotor(
                    n_blades=individual.B,
                    diameter=individual.D,
                    hub_radius=individual.hub_radius,
                    number_of_sections=self.number_of_sections,
                    foil_list=individual.foil_list,
                    chord_list=individual.chord_list,
                    pitch_list=individual.pitch_list
                )

                aerial = None
                aquatic = None
                
                try:
                    aT, aQ, aP, aJ, aCT, aCQ, aCP, aEta, FM = self.aerial_evaluation_method.evaluate(rotor)
                    
                    aQ_penalty = self._torque_penalty(aQ, self.aerial_Q_max)

                    if (aJ == 0):
                        # Use Figure of Merit in hover
                        individual.aerial_fitness = self._compute_aerial_fitness(FM, aQ_penalty)
                    else:
                        # Use efficiency otherwise
                        individual.aerial_fitness = self._compute_aerial_fitness(aEta, aQ_penalty)  

                    individual.aerial_fitness = self._safe_real(individual.aerial_fitness) # se ainda estiver gerando um complex type

                    aerial = {"T": aT, "Q": aQ, "P": aP, "J": aJ, "CT": aCT, "CQ": aCQ, "CP": aCP, "eta": aEta, "FM": FM, "Q_penalty": aQ_penalty}
                except Exception as e:
                    logger.warning(f"[NSGA] Aerial eval failed for individual (B={individual.B}, foils={set(individual.foil_list)}): {e}")
                    individual.aerial_fitness = 0.0  # penalization
                    aerial = None
                
                try:
                    wT, wQ, wP, wJ, wCT, wCQ, wCP, wEta, cavitating_proportion, QI = self.aquatic_evaluation_method.evaluate(rotor)
                    
                    individual.cavitating_proportion = cavitating_proportion
                
                    cavitation_penalty  = self._cavitation_penalty(cavitating_proportion)
                    wQ_penalty          = self._torque_penalty(wQ, self.aquatic_Q_max)

                    if (wJ == 0):
                        # Use Quality Index as fitness in bollard pull
                        individual.aquatic_fitness = self._compute_aquatic_fitness(QI, cavitating_proportion, wQ_penalty) 
                    else:
                        # Use efficiency otherwise
                        individual.aquatic_fitness = self._compute_aquatic_fitness(wEta, cavitating_proportion, wQ_penalty)

                    individual.aquatic_fitness = self._safe_real(individual.aquatic_fitness) # se ainda estiver gerando um complex type

                    aquatic = {"T": wT, "Q": wQ, "P": wP, "J": wJ, "CT": wCT, "CQ": wCQ, "CP": wCP, "eta": wEta, "cavitating_proportion": cavitating_proportion, "QI": QI, "cavitation_penalty": cavitation_penalty, "Q_penalty": wQ_penalty}
                except Exception as e:
                    logger.warning(f"[NSGA] Aquatic eval failed for individual (B={individual.B}, foils={set(individual.foil_list)}): {e}")
                    individual.aquatic_fitness = -1e6  # penalization
                    aquatic = None
                
                self._append_eval_row(
                    generation=generation,
                    pop_index=idx,
                    individual=individual,
                    aerial=aerial,
                    aquatic=aquatic
                )

            # Perform non-dominated sorting
            fronts = self._fast_non_dominated_sort(population)

            # If last generation, dont't create next generation
            if (generation == self.maximum_generations):
                logger.info("Maximum generations reached, stopping.")
                break

            # Calculate crowding distance for each front
            for front in fronts:
                self._crowding_distance(front)

            # Selection, crossover, and mutation
            population = self._create_next_generation(population, fronts)
        
        if fronts and len(fronts[0]) > 0:
            self._write_pareto_front_csv(fronts[0])
            logger.info(f"[NSGA-II] Pareto front salvo em {self.front_csv_path}")

        return fronts           
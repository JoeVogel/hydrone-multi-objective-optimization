import os
import random
import logging
import csv
import math
import copy

import pandas as pd

from datetime import datetime
from pathlib import Path

from evaluation.evaluation_method   import EvaluationMethod
from .individual                    import Individual
from rotor                          import Rotor

logger = logging.getLogger(__name__)

"""
The NSGA-II algorithm is a popular genetic algorithm for solving multi-objective optimization problems. 
It was first proposed by Deb et al. in 2002 as an improvement over the original Non-dominated Sorting Genetic Algorithm (NSGA).
https://ieeexplore.ieee.org/abstract/document/996017?casa_token=dDGNqrFAX2YAAAAA:zLPH-VzJoi06EqWav3xZuAB8pgTXSV7SsAm7sLSv9yU4g_-VStrzhtsJjnDwe6hTkMeuXAf1nA
"""

# TODO: avaliar implementação exemplo:
# https://github.com/smkalami/nsga2-in-python/blob/main/nsga2.py

class NSGAII:
    def __init__(self, aerial_evaluation_method: EvaluationMethod, aquatic_evaluation_method: EvaluationMethod, problem_configuration, nsga_configuration, write_log_file=True):
        
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

        self.aerial_sigma_root = 0.40
        self.aerial_sigma_tip  = 0.12

        self.aquatic_sigma_root = 0.70
        self.aquatic_sigma_tip  = 0.25

        self.hub_diameter = self.hub_radius * 2 

        self._write_log_file = write_log_file

        self._init_run_outputs()

        logger.info(f"[NSGA-II] Population size: {self.population_size}, Generations: {self.maximum_generations}, Seed: {self.seed}")

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
            "aerial_T","aerial_Q","aerial_P","aerial_J","aerial_CT","aerial_CQ","aerial_CP","aerial_eta", "FM", "aerial_fitness", "aerial_solidity_penalty", "aerial_blade_count_penalty",
            # water metrics
            "aquatic_T","aquatic_Q","aquatic_P","aquatic_J","aquatic_CT","aquatic_CQ","aquatic_CP","aquatic_eta", "cavitating_proportion", "QI","aquatic_fitness", "cavitation_penalty", "aquatic_solidity_penalty",
        ]

        if self._write_log_file:
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
    def _safe_max(values, default=None):
        """Returns the maximum of a list of values, safely ignoring None and NaN. Returns default if no valid values."""
        vals = []
        for v in values:
            if v is None:
                continue
            v = NSGAII._safe_real(v)
            if v is None:
                continue
            if isinstance(v, (int, float)) and (v == v):  # filtra NaN
                vals.append(float(v))
        return max(vals) if vals else default

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
            "aerial_solidity_penalty": "" if aerial  is None else aerial.get("solidity_penalty",""),
            "aerial_blade_count_penalty": "" if aerial  is None else aerial.get("blade_count_penalty",""),
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
            "cavitation_penalty": "" if aquatic is None else aquatic.get("cavitation_penalty",""),
            "aquatic_solidity_penalty": "" if aquatic is None else aquatic.get("solidity_penalty",""),
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

            a_fitness, a = self._aerial_eval(rotor)
            ind.aerial_fitness = a_fitness

            w_fitness, w = self._aquatic_eval(rotor)
            ind.aquatic_fitness = w_fitness
            ind.cavitating_proportion = w.get("cavitating_proportion", None) if w is not None else None

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

            max_delta_pitch = (self.max_alpha - self.min_alpha) * 0.2  # Exemple: 20% of range
            max_delta_chord = 0.005 # Exemple: 0.5 cm

            pitch = self._smooth_profile(pitch, max_delta_pitch)
            chord = self._smooth_profile(chord, max_delta_chord)
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

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two parents to produce a child.
        """

        n = self.number_of_sections
        cut = random.randint(1, n - 1)

        child_pitch = list(parent1.pitch_list[:cut]) + list(parent2.pitch_list[cut:])
        child_chord = list(parent1.chord_list[:cut]) + list(parent2.chord_list[cut:])

        max_delta_pitch = (self.max_alpha - self.min_alpha) * 0.2  # Exemple: 20% of range
        max_delta_chord = 0.005 # Exemple: 0.5 cm

        self._smooth_profile(child_pitch, max_delta=max_delta_pitch)  # Limita variação de pitch entre seções
        self._smooth_profile(child_chord, max_delta=max_delta_chord)  # Limita variação de chord entre seções

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

        # Update individual
        individual.pitch_list = pitch_list
        individual.chord_list = chord_list
        individual.foil_list  = foil_list

        individual.aerial_fitness  = None
        individual.aquatic_fitness = None
        individual.cavitating_proportion = None

    def _cavitation_penalty(
            self, 
            cav: float,
            gamma: float = 1.0,
    ) -> float:
        """
        Calculates a penalty based on the cavitating proportion.

        param cav: Cavitating proportion (0 to 1)
        param gamma: Exponent to control penalty growth 
            1 = linear, 
            >1 = convex, more severe in higher cavitation,
            <1 = concave, more severe in lower cavitation)
        """
        cav = max(0.0, min(1.0, cav))  # Clamp to [0, 1]
        penalty = cav ** gamma
        return penalty

    def _solidity_penalty(
            self, 
            rotor,
            min_r_frac: float = 0.20,   # minimum radial fraction to start checking (e.g., 0.10)
            sigma_root: float = 0.24,   # sigma_max at x = min_r_frac (e.g., 0.20)
            sigma_tip: float  = 0.14,   # sigma_max at x = 1.0 
            # Normalization / saturation controls
            tau_mean: float = 0.10,     # mean relative violation that maps to penalty=1 (before mixing)
            tau_max: float  = 0.20,     # max relative violation that maps to penalty=1 (before mixing)
            beta: float = 0.70,         # mix weight: mean vs max (0..1)
            weight_power: float = 2.0,  # w_i = x^weight_power (emphasize tip)
            eps: float = 1e-12
    ) -> float:
        secs = getattr(rotor, "sections", None)
        if not secs:
            return 0.0
        
        R = float(getattr(rotor, "blade_radius", 0.0))
        if R <= 0.0:
            return 0.0
        
        B = int(getattr(rotor, "n_blades", 0))
        if B <= 0:
            return 0.0
        
        l_denom = max(eps, (1.0 - min_r_frac))
        slope = (sigma_tip - sigma_root) / l_denom
        
        h_denom = (1.0 / min_r_frac) - 1.0
        
        def sigma_max(x: float) -> float:
            # clamp x to [min_r_frac, 1.0]
            x = max(min_r_frac, min(1.0, x))

            # Linear interpolation between (min_r_frac, sigma_root) and (1.0, sigma_tip)
            # return max(eps, sigma_root + slope * (x - min_r_frac)) # linear
            
            # Hyperbolic interpolation
            num = (1.0 / x) - 1.0
            return sigma_tip + (sigma_root - sigma_tip) * (num / h_denom)  # adjusted hyperbolic
        
        violations = []
        weights = []

        for sec in secs:
            r  = float(getattr(sec, "radius", 0.0))
            if r <= 0.0:
                continue

            x = r / R
            if x < min_r_frac or x > 1.0:
                continue

            sigma_i = float(getattr(sec, "sigma"))
            sigma_i_max = sigma_max(x)

            # Relative violation (0 if within limit, >0 if violated)
            rel_violation = max(0.0, (sigma_i - sigma_i_max) / sigma_i_max)

            # Weight to emphasize tip (0 if within limit, >0 if violated)
            weight = x ** weight_power

            violations.append(rel_violation)
            weights.append(weight)

        if not violations:
            return 0.0
        
        # Weighted mean violation
        weighted_sum = sum(weights) + eps
        mean_violation = sum(v * w for v, w in zip(violations, weights)) / weighted_sum

        # Weighted max violation
        max_violation = max(violations)

        # Map to penalty [0,1]
        mean_penalty = min(1.0, mean_violation / max(eps, tau_mean))
        max_penalty  = min(1.0, max_violation  / max(eps, tau_max))

        # Combined penalty
        penalty = beta * mean_penalty + (1.0 - beta) * max_penalty
        return max(0.0, min(1.0, penalty))

    def _blade_count_penalty(
            self, 
            B: int,
            B_max: int,
            gamma: float = 2.0
        ) -> float:
        """
        Calculates a penalty based on the number of blades.
        """
        if B == 2:
            return 0.0
        
        denominator = max(1, (B_max - 2))
        penalty = ((B - 2) / denominator) ** gamma

        return max(0.0, min(1.0, penalty))

    def _compute_aerial_fitness(self, raw_result, solidity_penalty, blade_count_penalty):
        """
        Computes aerial fitness based on efficiency and penalties.
        """
        solidity_alpha = 0.7  # weight of solidity penalty
        blade_count_alpha = 0.7  # weight of blade count penalty

        fitness = raw_result * (1.0 - solidity_alpha * solidity_penalty) * (1.0 - blade_count_alpha * blade_count_penalty)
        return fitness

    def _compute_aquatic_fitness(self, raw_result, cavitation_penalty, solidity_penalty):
        """
        Computes aquatic fitness based on efficiency, cavitation penalty, and torque penalty.
        """
        cavitation_alpha = 1.0  # weight of cavitation penalty
        solidity_alpha = 0.2  # weight of solidity penalty, low in aquatic to prioritize cavitation

        fitness = raw_result * (1.0 - cavitation_alpha * cavitation_penalty) * (1.0 - solidity_alpha * solidity_penalty)
        return fitness

    def _aerial_eval(self, rotor):
        """
        Aerial evaluation for a given rotor.
        """
        try:
            aT, aQ, aP, aJ, aCT, aCQ, aCP, aEta, FM = self.aerial_evaluation_method.evaluate(rotor)

            solidity_penalty = self._solidity_penalty(rotor, sigma_root=0.40, sigma_tip=0.12)
            blade_count_penalty = self._blade_count_penalty(rotor.n_blades, self.max_blade_number)

            if (aJ == 0):
                # Use Figure of Merit in hover
                a_fitness = FM
            else:
                # Use efficiency otherwise
                a_fitness = aEta

            fitness = self._compute_aerial_fitness(a_fitness, solidity_penalty, blade_count_penalty)
            fitness = self._safe_real(fitness)

            return fitness, {"T": aT, "Q": aQ, "P": aP, "J": aJ, "CT": aCT, "CQ": aCQ, "CP": aCP, "eta": aEta, "FM": FM, "solidity_penalty": solidity_penalty, "blade_count_penalty": blade_count_penalty}
        except Exception as e:
            logger.warning(f"[NSGA] Aerial eval failed: {e}")
            fitness = 0.0  # penalization
            return fitness, None

    def _aquatic_eval(self, rotor):
        """
        Aquatic evaluation for a given rotor.
        """

        try:
            wT, wQ, wP, wJ, wCT, wCQ, wCP, wEta, cavitating_proportion, QI = self.aquatic_evaluation_method.evaluate(rotor)
                    
            cavitation_penalty  = self._cavitation_penalty(cavitating_proportion)
            solidity_penalty = self._solidity_penalty(rotor, sigma_root=0.70, sigma_tip=0.25)

            if (wJ == 0):
                # Use Quality Index as fitness in bollard pull
                w_fitness = QI 
            else:
                # Use efficiency otherwise
                w_fitness = wEta

            fitness = self._compute_aquatic_fitness(w_fitness, cavitation_penalty, solidity_penalty)
            fitness = self._safe_real(fitness)

            return fitness, {"T": wT, "Q": wQ, "P": wP, "J": wJ, "CT": wCT, "CQ": wCQ, "CP": wCP, "eta": wEta, "cavitating_proportion": cavitating_proportion, "QI": QI, "cavitation_penalty": cavitation_penalty, "solidity_penalty": solidity_penalty}
        except Exception as e:
            logger.warning(f"[NSGA] Aquatic eval failed for individual: {e}")
            fitness = 0.0  # penalization
            return fitness, None

    def run(self):
        """
        Executes the NSGA-II optimization process and returns Pareto fronts.
        """

        bests_rows = []

        population = self._initialize_population()
        for generation in range(1, self.maximum_generations + 1):
            logger.info(f"Generation {generation}/{self.maximum_generations}")

            # Evaluate population
            for idx, individual in enumerate(population):

                logger.info(
                    f"[NSGA] Evaluating individual {idx+1}/{len(population)} Gen {generation}/{self.maximum_generations} "
                    f"(B={individual.B}, "
                    f"unique_foils={len(set(individual.foil_list))})"
                )

                rotor = Rotor(
                    n_blades=individual.B,
                    diameter=individual.D,
                    hub_radius=individual.hub_radius,
                    number_of_sections=self.number_of_sections,
                    foil_list=individual.foil_list,
                    chord_list=individual.chord_list,
                    pitch_list=individual.pitch_list
                )

                aerial_fitness, aerial = self._aerial_eval(rotor)
                individual.aerial_fitness = aerial_fitness

                aquatic_fitness, aquatic = self._aquatic_eval(rotor)
                individual.cavitating_proportion = aquatic.get("cavitating_proportion", None) if aquatic is not None else None
                individual.aquatic_fitness = aquatic_fitness

                if (self._write_log_file):
                    self._append_eval_row(
                        generation=generation,
                        pop_index=idx,
                        individual=individual,
                        aerial=aerial,
                        aquatic=aquatic
                    )

            best_aerial = self._safe_max((ind.aerial_fitness for ind in population), default=None)
            best_aquatic = self._safe_max((ind.aquatic_fitness for ind in population), default=None)

            bests_rows.append({
                "generation": generation,
                "best_aerial_fitness": best_aerial,
                "best_aquatic_fitness": best_aquatic,
            })

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

        df_bests_by_generation = pd.DataFrame(bests_rows)

        out_path = self.run_dir / "bests_by_generation.csv"
        df_bests_by_generation.to_csv(out_path, index=False)
        logger.info(f"[NSGA-II] Bests by generation saved in {out_path}")

        if fronts and len(fronts[0]) > 0:
            self._write_pareto_front_csv(fronts[0])
            logger.info(f"[NSGA-II] Pareto front saved in {self.front_csv_path}")

        return fronts           
import os
import random
import logging
import csv

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
    def __init__(self, population_size, maximum_generations, aerial_evaluation_method: EvaluationMethod, aquatic_evaluation_method: EvaluationMethod, seed=0, mutation_rate=0.1):
        self.population_size            = population_size
        self.maximum_generations        = maximum_generations
        self.aerial_evaluation_method   = aerial_evaluation_method
        self.aquatic_evaluation_method  = aquatic_evaluation_method
        self.seed                       = seed
        self.mutation_rate              = mutation_rate
        
        csv_path = os.path.join(os.path.dirname(__file__), "../../data/decision_variables.csv")
        df = pd.read_csv(csv_path)
        
        self.min_alpha = df["min_alpha"][0]
        self.max_alpha = df["max_alpha"][0]
        self.diameter = df["diameter"][0]
        self.min_blade_number = df["min_blade_num"][0]
        self.max_blade_number = df["max_blade_num"][0]
        self.foil = df["foil"][0]
        self.number_of_sections = df["number_of_sections"][0]

        # TODO: definir chord_list e radius_hub como parâmetros de entrada ou calcular dinamicamente
        self.chord_list = [
            0.01700, 0.01660, 0.01620, 0.01580, 0.01540,
            0.01500, 0.01440, 0.01380, 0.01320, 0.01260,
            0.01220, 0.01180, 0.01140, 0.01100, 0.01060,
            0.01020, 0.00980, 0.00940, 0.00900, 0.00860
            ]
        self.radius_hub = 0.0025

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
            "D", "B", "radius_hub", "number_of_sections",
            "foil_list", "chord_list", "pitch_list",
            # Scenario (for traceability)
            "aerial_rpm", "aerial_v_inf", "aquatic_rpm", "aquatic_v_inf",
            # aerial metrics
            "aerial_T","aerial_Q","aerial_P","aerial_J","aerial_CT","aerial_CQ","aerial_CP","aerial_eta", "FM", "aerial_fitness",
            # water metrics
            "aquatic_T","aquatic_Q","aquatic_P","aquatic_J","aquatic_CT","aquatic_CQ","aquatic_CP","aquatic_eta", "cavitating_proportion", "QI","aquatic_fitness",
        ]

        with self.eval_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._eval_fields)
            writer.writeheader()

        logger.info(f"[NSGA-II] Result dir: {self.run_dir}")

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
            "radius_hub": getattr(individual, "radius_hub", ""),
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
                    radius_hub=ind.radius_hub,
                    number_of_sections=getattr(ind, "number_of_sections", self.number_of_sections),
                    foil_list=getattr(ind, "foil_list", None),
                    chord_list=getattr(ind, "chord_list", None),
                    pitch_list=getattr(ind, "pitch_list", None),
                )

                a = w = None
                try:
                    aT, aQ, aP, aJ, aCT, aCQ, aCP, aEta = self.aerial_evaluation_method.evaluate(rotor)
                    a = {"T": aT, "Q": aQ, "P": aP, "J": aJ, "CT": aCT, "CQ": aCQ, "CP": aCP, "eta": aEta}
                except Exception as e:
                    logger.debug(f"[Front CSV] Aerial eval failed: {e}")
                try:
                    wT, wQ, wP, wJ, wCT, wCQ, wCP, wEta, cavitating_proportion, QI = self.aquatic_evaluation_method.evaluate(rotor)
                    w = {"T": wT, "Q": wQ, "P": wP, "J": wJ, "CT": wCT, "CQ": wCQ, "CP": wCP, "eta": wEta, "cavitating_proportion": cavitating_proportion, "QI": QI}
                except Exception as e:
                    logger.debug(f"[Front CSV] Aquatic eval failed: {e}")

                self._append_eval_row(
                    generation=self.maximum_generations,  # último gen
                    pop_index=i,
                    individual=ind,
                    aerial=a,
                    aquatic=w,
                    target_path=self.front_csv_path
                )

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
                    radius_hub=individual.radius_hub,
                    number_of_sections=self.number_of_sections,
                    foil_list=individual.foil_list,
                    chord_list=individual.chord_list,
                    pitch_list=individual.pitch_list
                )

                aerial = None
                aquatic = None
                
                try:
                    aT, aQ, aP, aJ, aCT, aCQ, aCP, aEta, FM = self.aerial_evaluation_method.evaluate(rotor)

                    if (aJ == 0):
                        individual.aerial_fitness = FM  # Usar Figure of Merit como fitness em condição de hover
                    else:
                        individual.aerial_fitness = aEta  # Usar eficiência normal em outras condições
                        
                    aerial = {"T": aT, "Q": aQ, "P": aP, "J": aJ, "CT": aCT, "CQ": aCQ, "CP": aCP, "eta": aEta, "FM": FM}
                except Exception as e:
                    logger.debug(f"[NSGA] Aerial eval failed for individual: {e}")
                    individual.aerial_fitness = 0.0  # penalização
                
                try:
                    wT, wQ, wP, wJ, wCT, wCQ, wCP, wEta, cavitating_proportion, QI = self.aquatic_evaluation_method.evaluate(rotor)

                    # TODO: definir aplicação de penalidade por cavitação
                    
                    if (wJ == 0):
                        individual.aquatic_fitness = QI  # Usar Quality Index como fitness em condição de amarra
                    else:
                        individual.aquatic_fitness = wEta  # Usar eficiência normal em outras condições

                    aquatic = {"T": wT, "Q": wQ, "P": wP, "J": wJ, "CT": wCT, "CQ": wCQ, "CP": wCP, "eta": wEta, "cavitating_proportion": cavitating_proportion, "QI": QI}
                except Exception as e:
                    logger.debug(f"[NSGA] Aquatic eval failed for individual: {e}")
                    individual.aquatic_fitness = 0.0  # penalização
                
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

    def _initialize_population(self):
        """
        Initializes a random population of individuals.
        """
        population = []
        
        random.seed(self.seed)
        
        for _ in range(self.population_size):

            individual = Individual(
                alpha=round(random.uniform(self.min_alpha, self.max_alpha), 1), # round para ter apenas 1 casa decimal
                D=self.diameter,
                B=random.randint(self.min_blade_number, self.max_blade_number),
                chord_list=self.chord_list,  
                foil=self.foil,
                radius_hub=self.radius_hub,  
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
        """
        next_generation = []
        i = 0

        # Adicionar as frentes até que a população atinja o tamanho necessário
        while i < len(fronts) and len(next_generation) + len(fronts[i]) <= self.population_size:
            next_generation.extend(fronts[i])  # Adiciona a frente inteira
            i += 1

        # Se ainda falta espaço, ordenamos por crowding distance e completamos
        remaining_spots = self.population_size - len(next_generation)
        if remaining_spots > 0:
            fronts[i].sort(key=lambda x: -x.crowding_distance)  # Ordem decrescente de crowding distance
            next_generation.extend(fronts[i][:remaining_spots])  # Preenche com os melhores

        # Aplicar torneio + crossover + mutação para gerar filhos até preencher a população
        while len(next_generation) < self.population_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover e mutação
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

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two parents to produce a child.
        """

        child = Individual(
            alpha=(parent1.alpha + parent2.alpha) / 2,
            D=self.diameter,
            B=int((parent1.B + parent2.B) / 2),
            chord_list=self.chord_list, # Assuming chord_list is the same for every individual
            foil=self.foil, # Assuming chord_list is the same for every individual
            radius_hub=self.radius_hub, # Assuming chord_list is the same for every individual
            number_of_sections=self.number_of_sections # Assuming chord_list is the same for every individual
        )

        return child

    def _mutate(self, individual, mutation_rate=0.1):
        """
        Perform mutation on an individual's parameters.
        """
        # Aplicação da mutação com restrição de valores dentro dos limites
        if random.random() < mutation_rate:
            individual.alpha += random.uniform(-0.1, 0.1)
            individual.alpha = max(self.min_alpha, min(individual.alpha, self.max_alpha))  # Garante que alpha esteja dentro dos limites

        if random.random() < mutation_rate:
            individual.B += random.randint(-1, 1)  # Como B é inteiro, usar randint é melhor
            individual.B = max(self.min_blade_number, min(individual.B, self.max_blade_number))  # Garante que B esteja dentro dos limites
            